import argparse
import numpy as np
import os, cv2
from tqdm import tqdm
import paddle
import time
import paddle.nn.functional as F

####################################################
NET_NAME = 'AIDR'
####################################################

def read_img(path, size=None):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

def pd_tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    img = tensor.squeeze().cpu().numpy()
    img = img.clip(min_max[0], min_max[1])
    img = (img - min_max[0]) / (min_max[1] - min_max[0])
    if out_type == np.uint8:
        # scaling
        img = img * 255.0
    img = np.transpose(img, (1, 2, 0))
    img = img.round()
    img = img[:,:,::-1]
    return img.astype(out_type)

def load_network(weight_path, network):
    print('Loading checkpoint from: {}'.format(weight_path))
    weights = paddle.load(os.path.join(weight_path))
    network.load_dict(weights)

def uint2tensor4(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img = img.astype(np.float32) / 255.
    return paddle.to_tensor(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).unsqueeze(0)


def main(args):
    # set device
    if paddle.is_compiled_with_cuda():
        paddle.set_device('gpu:0')
    else:
        paddle.set_device('cpu')

    # make dirs
    os.makedirs(os.path.join(args.save_path), exist_ok=True)
    # set model
    if NET_NAME == 'AIDR':
        from modules.AIDR_arch import AIDR
        net = AIDR(num_c=96)

    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(NET_NAME))
    print('=========> Load From: ', args.pretrained)
    load_network(args.pretrained, net)
    net.eval()

    out_dir = os.path.join(args.save_path)

    TIME= []
    for img_name in tqdm(os.listdir(os.path.join(args.root_test))):
        img = read_img(os.path.join(args.root_test, img_name))
        img = img[:, :, [2, 1, 0]]  # bgr -> rgb
        img = img.astype(np.float32)
        input_tensor = paddle.to_tensor(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).unsqueeze(0)
        _, _, h, w = input_tensor.shape

        pad_size = 128
        h_padded = False
        w_padded = False
        if h % pad_size != 0:
            pad_h = pad_size - (h % pad_size)
            input_tensor = F.pad(input_tensor, (0, 0, 0, pad_h), mode='reflect')
            h_padded = True

        if w % pad_size != 0:
            pad_w = pad_size - (w % pad_size)
            input_tensor = F.pad(input_tensor, (0, pad_w, 0, 0), mode='reflect')
            w_padded = True

        # print(input_tensor.shape)
        t0 = time.time()
        with paddle.no_grad():
            result_tensor = net(input_tensor)
            if args.test_enhance:
                result_tensor += paddle.flip(net(paddle.flip(input_tensor, axis=[2])), axis=[2])
                result_tensor += paddle.flip(net(paddle.flip(input_tensor, axis=[3])), axis=[3])
                result_tensor += paddle.flip(net(paddle.flip(input_tensor, axis=[2, 3])), axis=[2, 3])
                result_tensor = result_tensor / 4
        TIME.append(time.time() - t0)
        # remove extra pad
        if h_padded:
            result_tensor = result_tensor[:, :, 0:h, :]
        if w_padded:
            result_tensor = result_tensor[:, :, :, 0:w]
        result_img = pd_tensor2img(result_tensor) # 已转bgr
        cv2.imwrite(out_dir + '/{}'.format(img_name), result_img)


if __name__== '__main__':
    parser = argparse.ArgumentParser(description="BaiDu")
    parser.add_argument("--root_test", type=str, default="./dataset/baidu/task1/moire_testB_dataset/",
                        help='dataset directory')
    parser.add_argument("--save_path", type=str, default="./result_testBx4", help='dataset directory')
    parser.add_argument("--pretrained", default="./checkpoint/best.pdparams", type=str, help="path to pretrained models")
    parser.add_argument("--test_psnr", default=False, type=bool, help="test psnr")
    parser.add_argument("--test_enhance", default=True, type=bool, help="test enhance")
    args = parser.parse_args()
    main(args)



