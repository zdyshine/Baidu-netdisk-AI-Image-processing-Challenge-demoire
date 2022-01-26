import argparse
import numpy as np
from tqdm import tqdm
from dataloading import dataloader_task1
from loss.msssim import MS_SSIMLoss
import logging
import random
import utils
import os

import paddle
import paddle.nn as nn
####################################################
NET_NAME = 'AIDR'
####################################################

def get_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        print('param_group',param_group['lr'])
        param_group['lr'] = lr

def load_network(weight_path, network):
    print('Loading checkpoint from: {}'.format(weight_path))
    weights = paddle.load(os.path.join(weight_path))
    network.load_dict(weights)

def main(args):
    # set log
    log_file = os.path.join('./log', NET_NAME + '_log_576.txt')
    if args.loss_msssim:
        log_file = os.path.join('./log', NET_NAME + '_log_ms.txt')
    logging = utils.setup_logger(output=log_file, name=NET_NAME)
    logging.info(args)

    # set gpu
    if paddle.is_compiled_with_cuda():
        paddle.set_device('gpu:0')
    else:
        paddle.set_device('cpu')

    # set random seed
    logging.info('========> Random Seed: {}'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    paddle.framework.random._manual_program_seed(args.seed)

    # set model
    if NET_NAME == 'AIDR':
        from modules.AIDR_arch import AIDR
        net = AIDR(num_c=96)#96
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(NET_NAME))
    logging.info('=========> net name: {}'.format(NET_NAME))

    # evaluate or finute, only load generator weights
    if args.pretrained:
        logging.info('=========> Load From: {}'.format(args.pretrained))
        load_network(args.pretrained, net)
    print_network(net)

    # set dataload
    train_data = dataloader_task1.Task1_Traindataset(args, mode='train')
    train_loader = paddle.io.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_data = dataloader_task1.Task1_Traindataset(args, mode='val')
    val_loader = paddle.io.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2)

    # set optimizer
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=args.lr, step_size=args.lr_decay_iters, gamma=args.gamma, verbose=False)
    optimizer = paddle.optimizer.Adam(scheduler, parameters=net.parameters(), weight_decay=0.0)

    loss_criterion = nn.L1Loss()
    loss_msssim = MS_SSIMLoss(window_size=3, data_range=1)
    iters = 0
    best_score = -1.0
    net.train()
    logging.info('=========> Start Train ...')
    for epoch in range(1, args.epochs + 1):
        train_loss = 0
        train_loss_l1 = 0
        train_loss_ssim = 0

        for i, (tensor_damage, tensor_ref) in enumerate(train_loader):
            # print(tensor_damage.shape, tensor_ref.shape)
            iters += 1
            optimizer.clear_grad()
            out_tensor = net(tensor_damage)
            if args.debug:
                print(tensor_damage.shape, tensor_ref.shape, out_tensor.shape)
            # loss_l1 = loss_criterion(out_tensor, tensor_ref)
            loss_l1_r = loss_criterion(out_tensor[0, :, :], tensor_ref[0, :, :])
            loss_l1_g = loss_criterion(out_tensor[1, :, :], tensor_ref[1, :, :])
            loss_l1_b = loss_criterion(out_tensor[2, :, :], tensor_ref[2, :, :])
            loss_l1 = loss_l1_r + loss_l1_g + loss_l1_b

            loss = loss_l1
            if args.loss_msssim:
                loss_ms = loss_msssim(out_tensor, tensor_ref)
                loss = loss / 100. + loss_ms

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            train_loss_l1 += loss_l1.item()
            if args.loss_msssim:
                train_loss_ssim += loss_ms.item()
            else: 
                train_loss_ssim = 0
            if iters % (args.iter_print) == 0: # 200 iter 打印一次
                log_info = 'epoch:{}, Total iter:{}, iter:{}|{}, TLoss:{:.4f}, L1Loss:{:.4f}, SLoss:{:.6f}. Lr:{}'.format(
                    epoch, iters, i, len(train_loader), train_loss / (i + 1), train_loss_l1 / (i + 1), train_loss_ssim / (i + 1), optimizer.get_lr()
                )
                # print(log_info)
                logging.info(log_info)

            if iters % args.iter_val == 0:
                # val and save
                avg_psnr, avg_msssim = validate(val_loader, net, loss_msssim)
                score = (avg_psnr / 100 + avg_msssim) * 0.5
                is_best = score > best_score
                best_score = max(score, best_score)
                save_checkpoint(iters, net, score)
                if is_best:
                    save_best_checkpoint(net)
                logging.info("===> Valid. psnr: {:.4f}, msssim: {:.4f}, score: {:.4f}, Best score: {:.4f}".format(avg_psnr, avg_msssim, score, best_score))

def validate(val_loader, net, loss_msssim):
    net.eval()
    avg_psnr = 0
    avg_msssim = 0

    with paddle.no_grad():
        for i, (tensor_damage, tensor_ref) in tqdm(enumerate(val_loader)):
            tensor_damage = tensor_damage[:,:,10:522,10:522]
            tensor_ref = tensor_ref[:,:,10:522,10:522]
            out_tensor = net(tensor_damage)
            ms_ssim_value = loss_msssim.mssim_value(out_tensor, tensor_ref)
            avg_msssim += ms_ssim_value.item()
            output = utils.pd_tensor2img(out_tensor)
            target = utils.pd_tensor2img(tensor_ref)
            pnsr_value = utils.compute_psnr(target, output)
            avg_psnr += pnsr_value
    net.train()
    return avg_psnr / len(val_loader), avg_msssim / len(val_loader)


def save_checkpoint(iters, model, psnr):
    output_dir = os.path.join('checkpoint', NET_NAME)
    os.makedirs(output_dir, exist_ok=True)

    save_filename = 'epoch_{}_{:.4f}.pdparams'.format(iters, psnr)
    save_path = os.path.join(output_dir, save_filename)
    logging.info('Saving checkpoint to: {}\n'.format(save_path))
    paddle.save(model.state_dict(), save_path)

def save_best_checkpoint(model):
    output_dir = os.path.join('checkpoint', NET_NAME)
    os.makedirs(output_dir, exist_ok=True)

    save_filename = 'best.pdparams'
    save_path = os.path.join(output_dir, save_filename)
    print('Saving checkpoint to: {}\n'.format(save_path))
    logging.info('Saving checkpoint to: {}\n'.format(save_path))
    paddle.save(model.state_dict(), save_path)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    logging.info('Total number of parameters: %d' % num_params)

if __name__== '__main__':
    parser = argparse.ArgumentParser(description="BaiDu")
    # Model Selection
    parser.add_argument('--net_name', type=str, default=NET_NAME)
    # Hardware Setting
    parser.add_argument('--seed', type=int, default=2021, help='Random seed')

    # Directory Setting
    parser.add_argument('--train_root', type=str, default='./dataset/baidu/task1/moire_train_dataset')
    parser.add_argument('--val_root', type=str, default='./dataset/baidu/task1/moire_val_dataset')
    parser.add_argument("--use_flip", type=bool, default=True)
    parser.add_argument("--use_rot", type=bool, default=True)
    parser.add_argument('--patch_size', type=int, default=512, help='target size') # 256
    parser.add_argument("--pretrained", default="", type=str, help="path to pretrained models")
    parser.add_argument('--iter_print', type=int, default=10, help='learning rate decay per N iters')
    parser.add_argument('--iter_val', type=int, default=100, help='learning rate decay per N iters')

    # Learning Options
    parser.add_argument('--epochs', type=int, default=2000, help='Max Epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--loss', type=str, default='l1', help='loss function configuration')
    parser.add_argument('--loss_msssim', type=bool, default=False, help='loss function configuration')

    # Optimization specifications
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate') #  2e-4 -> 2e-5
    parser.add_argument('--lr_decay_iters', type=int, default=300000, help='learning rate decay per N iters')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='min learning rate')
    parser.add_argument('--decay_type', type=str, default='step', help='learning rate decay type')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
    parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop', 'ADAMax'),
                        help='optimizer to use (SGD | ADAM | RMSprop | ADAMax)')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument("--debug", type=bool, default=False)

    args = parser.parse_args()
    main(args)
