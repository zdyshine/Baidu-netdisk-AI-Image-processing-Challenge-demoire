# 百度网盘AI大赛——图像处理挑战赛：文档图像摩尔纹消除第2名方案
百度网盘AI大赛——图像处理挑战赛：文档图像摩尔纹消除第2名方案


# 去除摩尔纹，A榜第一，B榜第二。
采用改进的多尺度卷积神经网络来去除摩尔纹。
多尺度网络基于IDR: Self-Supervised Image Denoising via Iterative Data Refinement进行改动。

比赛连接：[百度网盘AI大赛：文档图像摩尔纹消除(赛题一)](https://aistudio.baidu.com/aistudio/competition/detail/128/0/task-definition)

# 一、任务分析
本次比赛去摩尔纹，通过消除拍摄时的摩尔纹噪声，还原图片原本的样子，其本质，也是图像复原任务。从评价指标来看，主要是PSNR和MSSSIM，也是图像复原任务中常用的客观评价指标，并且指标归一化后也是很合理的，PSNR和MSSSIM占比一样。其次，从本次的比赛训练数据和测试数据分析。训练数据有1000对，其中包含典型摩尔纹的场景数据占比较少，更多的数据场景，可以归为色彩增强。但是测试集A和测试集B中，典型摩尔纹场景的图片占比较大，因此在这批训练数据上训练的模型，在测试集中会有不稳定的差异。**解决方案来讲**，最直观有效的是增加对应的训练数据。其次就是进行后处理。

![](https://ai-studio-static-online.cdn.bcebos.com/123fbcfbcadb4718849ba59bf86da89dc01e6f428e9e4cddb52664758cdd8262)


# 二、模型构建思路及调优过程

（1）**算法思路**；    
基于对本次训练数据的分析，我们团队没有直接尝试经典的去摩尔纹的网络。而是修改了几个去噪网络来进行任务。最初是对去年性能很好的MIMOUNet，Restormer以及Uformer网络进行修改。其中MIMOUNet只进行三次降采样，训练和修改的效果来看，线上没有训练很高。。而基于Transformer的方案网络存在训练和测试，效果差异大，不稳定的情况，因此舍弃了使用Transformer的方案(这里不排除是我训练的不够好的情况，各位如果有训练效果还可以的，欢迎交流)。后来对比赛任务进一步分析，感觉无论是色彩在增强还是去摩尔纹，不同的图片的退化差异都比去噪任务更大，需要比去噪更大的感受野。因此就想尝试更多次的降采样，于是使用paddle实现了IDR网络。初次的训练线上分数0.66，提分很明显。之后在该网络的底层增加Non-Local模型，进一步增大感受野。指标也有了进一步的提升。    
    **主要改动：**    
    基于IDR网络：     
    1.在底层叠加了Non-Local模块，提高网络获取全局信息的能力。    
    2.把网络特征通道从48增加到96，提高网络的学习能力。
![](https://ai-studio-static-online.cdn.bcebos.com/1f5e9e385b1e44b2ae528ee9fceb465f4333b0102b694df095b97c037df72da3)

 
（2）数据增强/清洗策略；        
   1.训练数据增强：水平翻转，竖直翻转，旋转。    
   2.测试增强：水平翻转，竖直翻转。    

（3）调参优化策略；    
    每30w iter，lr减半

（4）训练脚本/代码    
    python train.py

（5）测试脚本/代码，必须包含评估得到最终精度的运行日志；
    python test.py
   

# 三、后处理流程：
![](https://ai-studio-static-online.cdn.bcebos.com/5afc6f65e6374efa84a35313f0f0bfdf58b272d03a1f4d08b59f7a401cbf12eb)

**说明：**测试集B更适合后处理，PSNR提升应该在1db，但是由于需要手动设置阈值，未使用该方式。

# 四、代码内容说明
checkpoint: 保存模型的文件夹    
dataloading: 定义数据加载    
modules: 定义模型    
log: 训练日志    
loss: 损失函数    

# 四、预训练模型：    
https://aistudio.baidu.com/aistudio/projectdetail/3439099    
运行项目，下载预训练模型，同时可以进行在线测试。
