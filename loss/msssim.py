import paddle
import paddle.nn.functional as F

'''

'''

def gaussian1d(window_size, sigma):
    ###window_size = 11
    x = paddle.arange(window_size,dtype='float32')
    x = x - window_size//2
    gauss = paddle.exp(-x ** 2 / float(2 * sigma ** 2))
    # print('gauss.size():', gauss.size())
    ### torch.Size([11])
    return gauss / gauss.sum()

def create_window(window_size, sigma, channel):
    _1D_window = gaussian1d(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    # print('2d',_2D_window.shape)
    # print(window_size, sigma, channel)
    return _2D_window.expand([channel,1,window_size,window_size])

def _ssim(img1, img2, window, window_size, channel=3 ,data_range = 255.,size_average=True,C=None):
    # size_average for different channel

    padding = window_size // 2

    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)
    # print(mu1.shape)
    # print(mu1[0,0])
    # print(mu1.mean())
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2
    if C ==None:
        C1 = (0.01*data_range) ** 2
        C2 = (0.03*data_range) ** 2
    else:
        C1 = (C[0]*data_range) ** 2
        C2 = (C[1]*data_range) ** 2
    # l = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    # ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    sc = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    lsc = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1))*sc

    if size_average:
        ### ssim_map.mean()是对这个tensor里面的所有的数值求平均
        return lsc.mean()
    else:
        # ## 返回各个channel的值
        return lsc.flatten(2).mean(-1),sc.flatten(2).mean(-1)

def ms_ssim(
    img1, img2, window, data_range=255, size_average=True, window_size=11, channel=3, sigma=1.5, weights=None, C=(0.01, 0.03)
):

    r""" interface of ms-ssim
    Args:
        img1 (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        img2 (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images should have the same dimensions.")

    # for d in range(len(img1.shape) - 1, 1, -1):
    #     img1 = img1.squeeze(dim=d)
    #     img2 = img2.squeeze(dim=d)

    if not img1.dtype == img2.dtype:
        raise ValueError("Input images should have the same dtype.")

    if len(img1.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(img1.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {img1.shape}")

    smaller_side = min(img1.shape[-2:])

    assert smaller_side > (window_size - 1) * (2 ** 4), "Image size should be larger than %d due to the 4 downsamplings " \
                                                        "with window_size %d in ms-ssim" % ((window_size - 1) * (2 ** 4),window_size)

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = paddle.to_tensor(weights)

    if window is None:
        window = create_window(window_size, sigma, channel)
    assert window.shape == [channel, 1, window_size, window_size], " window.shape error"

    levels = weights.shape[0] # 5
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs =  _ssim(img1, img2, window=window, window_size=window_size,
                                       channel=3, data_range=data_range,C=C, size_average=False)
        if i < levels - 1:
            mcs.append(F.relu(cs))
            padding = [s % 2 for s in img1.shape[2:]]
            img1 = avg_pool(img1, kernel_size=2, padding=padding)
            img2 = avg_pool(img2, kernel_size=2, padding=padding)

    ssim_per_channel = F.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = paddle.stack(mcs + [ssim_per_channel], axis=0)  # (level, batch, channel) 按照等级堆叠
    ms_ssim_val = paddle.prod(mcs_and_ssim ** weights.reshape([-1, 1, 1]), axis=0) # level 相乘
    # print(ms_ssim_val.shape)
    if size_average:
        return ms_ssim_val.mean()
    else:
        # 返回各个channel的值
        return ms_ssim_val.flatten(2).mean(1)


class SSIMLoss(paddle.nn.Layer):
   """
   1. 继承paddle.nn.Layer
   """
   def __init__(self, window_size=11, channel=3, data_range=255., sigma=1.5):
       """
       2. 构造函数根据自己的实际算法需求和使用需求进行参数定义即可
       """
       super(SSIMLoss, self).__init__()
       self.data_range = data_range
       self.C = [0.01, 0.03]
       self.window_size = window_size
       self.channel = channel
       self.sigma = sigma
       self.window = create_window(self.window_size, self.sigma, self.channel)
       # print(self.window_size,self.window.shape)
   def forward(self, input, label):
       """
       3. 实现forward函数，forward在调用时会传递两个参数：input和label
           - input：单个或批次训练数据经过模型前向计算输出结果
           - label：单个或批次训练数据对应的标签数据
           接口返回值是一个Tensor，根据自定义的逻辑加和或计算均值后的损失
       """
       # 使用Paddle中相关API自定义的计算逻辑
       # output = xxxxx
       # return output
       return 1-_ssim(input, label,data_range = self.data_range,
                      window = self.window, window_size=self.window_size, channel=3,
                      size_average=True,C=self.C)


# lossfn = SSIMLoss(window_size=3,data_range=1)
class MS_SSIMLoss(paddle.nn.Layer):
   """
   1. 继承paddle.nn.Layer
   """
   def __init__(self, data_range=255., channel=3, window_size=11, sigma=1.5):
       """
       2. 构造函数根据自己的实际算法需求和使用需求进行参数定义即可
       """
       super(MS_SSIMLoss, self).__init__()
       self.data_range = data_range
       self.C = [0.01, 0.03]
       self.window_size = window_size
       self.channel = channel
       self.sigma = sigma
       self.window = create_window(self.window_size, self.sigma, self.channel)
       # print(self.window_size,self.window.shape)
   def forward(self, input, label):
       """
       3. 实现forward函数，forward在调用时会传递两个参数：input和label
           - input：单个或批次训练数据经过模型前向计算输出结果
           - label：单个或批次训练数据对应的标签数据
           接口返回值是一个Tensor，根据自定义的逻辑加和或计算均值后的损失
       """
       # 使用Paddle中相关API自定义的计算逻辑
       # output = xxxxx
       # return output
       return 1-ms_ssim(input, label, data_range=self.data_range,
                      window = self.window, window_size=self.window_size, channel=self.channel,
                      size_average=True,  sigma=self.sigma,
                      weights=None, C=self.C)

   def mssim_value(self, input, label):
       return ms_ssim(input, label, data_range=self.data_range,
                      window = self.window, window_size=self.window_size, channel=self.channel,
                      size_average=True,  sigma=self.sigma,
                      weights=None, C=self.C)