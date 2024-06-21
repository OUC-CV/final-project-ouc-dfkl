# 项目报告

#### 视频网址：

## 绪论：

&emsp;&emsp;高动态范围成像（简称HDR），是提供比普通图像更多图像细节和动态范围的技术。HDR通过扩展图像的亮度和色彩范围呈现更多亮部与暗部的细节，还原更真实的视觉效果。<br>
&emsp;&emsp;本文基于《HDR image reconstruction from a single exposure using deep CNNs》进行组合创新。参考论文提出了基于全卷积神经网络（CNN）从低动态范围（LDR）输入图像中重建HDR图像。这个方法的形式为混合动态范围自编码器，通过编码器网络对LDR输入图像进行编码，然后将编码后的图像输入到HDR解码器网络，进行HDR图像的重建。同时，网络还配备了跳跃连接使得在重建时更好地利用高分辨率图像细节。参考论文为了训练CNN模型，通过收集大量HDR图像数据集来创建训练集，然后对于每个HDR图像使用虚拟相机模型模拟一组对应的LDR曝光图像。接着最小化自定义HDR损失函数来优化网络权重，并为了进一步提高鲁棒性，利用迁移学习在从MIT Places数据库子集创建的大量模拟HDR图像上预训练权重。该方法解决了在饱和图像区域中预测丢失信息的问题，以实现从单次曝光重建HDR。<br>
​&emsp;&emsp;本文基于参考论文的复现代码增加了低光照增强技术，去噪技术和SRCNN超分辨率技术进行组合创新。我们增加低光照增强技术以改善输入图像的光照条件，这种增强技术通过提升图像中低亮度区域的细节，使得HDR重建在这些区域的效果更加出色。我们使用去噪技术来降低输入图像中的噪声，从而提升HDR图像的清晰度和质量。此外，我们还使用SRCNN超分辨率技术来提升图像的细节，使得重建后的HDR图像在细节表现上更加丰富和真实。<br>

## 相关工作：
​&emsp;&emsp;SRCNN是超分辨率任务中最早且最具影响力的深度学习方法之一,由Dong等人提出[1]。超分辨率就是把低分辨率（LP）图像放大为高分辨率（HP）图像。SRCNN通过一个简单的三层卷积网络实现了这一功能。SRCNN的工作流程包括三步：首先，输入LP图像，经过双三次（bicubic）插值，被放大成目标尺寸，得到Y；然后，通过三层卷积网络拟合非线性映射；最后，输出HR图像结果F(Y)。<br>
​&emsp;&emsp;此后的研究工作在SRCNN的基础上进行了改进。FSRCNN[2]通过引入更深层次的网络结构和去除冗余计算，进一步提升了超分辨率性能和计算效率。VDSR[3]通过增加网络深度和引入残差学习，显著提高了超分辨率效果。EDSR[4]则通过移除批归一化层和进一步增加网络深度，达到了超分辨率任务的新高度。然而，SRCNN及其后续方法仍存在一些局限性。它们在处理极高放大倍数或图像内容复杂度较高的场景时，可能会出现伪影或细节丢失的问题。近年来，生成对抗网络（GAN）和其他深度学习方法在超分辨率任务中的应用也展示了更好的视觉效果和性能[5]。<br>
​&emsp;&emsp;Zero-Reference Deep Curve Estimation (Zero-DCE) 是一种用于低光照图像增强的基于卷积神经网络（CNN）架构的方法。该方法通过无监督的方式估计一组光线增强曲线，而不依赖任何参考图像。这种网络设计的目的是通过调整像素值来提高低光照条件下拍摄图像的可见度。低光照增强技术在深度学习的推动下取得了显著进展，但在真实数据的获取、计算资源的优化、时间一致性的处理以及模型效率方面仍有很多亟待解决的问题，这些问题的解决将进一步推动该领域的发展和应用。<br>

## 方法的具体细节：
#### 邓棋丹：
使用SRCNN超分辨率技术来提升图像的细节使得重建后的HDR图像在细节表现上更加丰富和真实。<br>
1.构建SRCNN模型<br>
SRCNN模型分为三部分：特征提取层、非线性映射层和网络重建层。<br>
特征提取层：通过CNN将图像Y的特征提取出来存于向量中。max(0,x)表示ReLU层。<br>
公式：<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/1.png)<br>
非线性映射层：把提取的特征做非线性映射，提高网络的深度和复杂度。<br>
公式：<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/2.png)<br>
重建层：借鉴传统超分的纯插值法做重建。这一层没有ReLU层。<br>
公式：<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/3.png)<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/4.png)

一开始，我运行super_resolutiontrain.py报错说输入张量与目标张量形状不一致。后面使用 torch.nn.functional.interpolate 函数对输出 x 进行插值操作，使得输入图像和输出图像形状一致。<br>
```python
class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = F.interpolate(x, size=(768, 1023), mode='bicubic', align_corners=False)
        return x

```
2.SRCNN图像颜色空间转换<br>
YCbCr颜色空间将颜色信息分成亮度（Y）和两个色度分量（Cb和Cr）。Y表示图像的亮度，包含图像的主要细节信息或灰度级别。Cb表示图像中的蓝色色度信息，即颜色的蓝色成分。Cr表示图像中的红色色度信息，即颜色的红色成分。<br>
在utils.py文件中添加convert_rgb_to_y，convert_rgb_to_ycbcr和convert_ycbcr_to_rgb三个函数。convert_rgb_to_y函数用于将RGB图像转换为亮度（Y）通道，convert_rgb_to_ycbcr函数用于将RGB图像转换为YCbCr颜色空间，convert_ycbcr_to_rgb函数用于将YCbCr图像转换回RGB颜色空间。<br>
从RGB到YCbCr公式：<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/5.png)
从YCbCr到RGB公式：<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/6.png)
3．应用SRCNN技术于HDR重建<br>
一开始，运行predict.py发现super_resolution函数无法正确地处理输入图像，显示图像没有image.width和image.height属性。原因是我输入的图像ldr是一个通过OpenCV加载的图像，它是NumPy 数组形式。我应该将其先转换为PIL图像对象：<br>
```python
#超分辨率
    ldr_pil = pil_image.fromarray(ldr)
    ldr_super_resolution = super_resolution(ldr_pil, 'weight/srcnn_x3.pth')
```
后面再次运行发现，ldr_super_resolution有图像，但是super_reconstructed是空白的，发现原来是因为ldr_super_resolution形状为(768*1023)，与(768*1024)不一致，通过插值算法调整图像的大小后解决了。<br>
```python
ldr_super_resolution_resized = cv2.resize(ldr_super_resolution, (1024, 768))
```
在ldr图像上使用超分辨率后进行HDR重建：<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/7.png)
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/8.png)


接着，想在经过低光照增强的图上使用超分辨率技术。发现ldr_enhance是一个 NumPy 数组和super_resolution函数输入图像类型不一致，要将其先转换为PIL图像对象，并且确保 ldr_enhance中的值在[0, 1]范围内。如果超出此范围，需要在传递给PIL前进行归一化处理。<br>
```python
image = pil_image.fromarray((ldr_enhance * 255).astype(np.uint8))
```

在ldr_enhance图像上使用超分辨率后进行HDR重建：<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/9.png)
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/10.png)


#### 范晓颖：
我们基于已有的从低动态范围(LDR)输入图像重建高动态范围(HDR)图像的方法基础上进行了组合式创新，我主要负责对原有代码进行改进以及增加低光照增强技术改善输入图像的光照条件，以便更好的进行hdr重建。<br>
在最开始我运行原有代码时发现运行效果不是很好，重建后的图像与输入图像差别不大，多次修改阈值后发现当tau参数较大时，重建图像与输入图像相同；tau参数较小时，重建图像呈现紫色。<br>
tau=0.95时：
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/11.png)
tau=0.05:
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/12.png)
经过调查发现:<br>
当 tau 较大时，计算得到的 alpha 值较小，意味着输入图像的平方项（input_image ** 2）的影响较小。这种情况下，重建图像与输入图像相似，因为 alpha 的影响较小，重建的颜色和亮度更接近原始输入。<br>
当 tau 较小时，计算得到的 alpha 值可能较大。这导致重建的 HDR 图像在混合过程中更多地受到输入图像平方项的影响。输入图像中较暗区域的平方项可能会导致颜色通道不平衡，尤其是在红色和蓝色通道上，这可能表现为图像呈现紫色。<br>
在发现问题的原因后我开始着手解决上述问题，我在计算重建图像之前添加了颜色通道的均衡处理功能。最开始我使用了简单的直方图均衡化方法来处理颜色通道。直方图均衡化代码：<br>
```python
def color_balance(image):
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab_image)
    L_eq = cv2.equalizeHist(L)
    lab_image_eq = cv2.merge((L_eq, A, B))
    balanced_image = cv2.cvtColor(lab_image_eq, cv2.COLOR_LAB2RGB)
    balanced_image = balanced_image / 255.0
    return balanced_image
```
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/13.png)
但是直方图均衡化可能会导致整体颜色偏移，特别是在亮度和对比度变化较大的图像上，最终得到的图像效果仍然不好。后来，我尝试使用更高级的方法——自适应直方图均衡化，它能更好地保持图像的整体色调，最终得到了效果较好的hdr重建图像。<br>
下面是改进后的 color_balance 函数，使用自适应直方图均衡化 (CLAHE) 来处理颜色通道：<br>
```python
def color_balance(image):
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    L, A, B = cv2.split(lab_image) # 分离 LAB 色彩空间的三个通道
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) # 创建 CLAHE 对象（自适应直方图均衡化）
    L_eq = clahe.apply(L)
    lab_image_eq = cv2.merge((L_eq, A, B))
    balanced_image = cv2.cvtColor(lab_image_eq, cv2.COLOR_LAB2RGB)
    
    balanced_image = balanced_image / 255.0
return balanced_image
```
得到的结果如下：<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/14.png)
原有代码经过改进后得到的训练和验证损失函数图像如下：<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/15.png)
为了改善视觉质量，更好地反映真实世界场景的光照变化，生成高质量HDR图像，我增加了低光照增强功能。该功能的添加我参考了论文《Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement》，将该论文提出的低光照增强方法组合到我们原有代码上，得到如下结果：<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/16.png)
可以看出，低光照增强后的图像相比输入图像更加明亮了，但过于明亮会导致hdr重建后的图像对比度降低，可能会造成颜色饱和度下降或色彩偏差，可能会导致图像中的高光区域细节丢失，影响HDR重建后的图像质量和视觉观感。<br>
因此需要降低光照增强的亮度，修改模型输出端的缩放因子来控制亮度增强强度：<br>
```python
        x = x + r1*(torch.pow(x,2)-x)
        x = x + r2*(torch.pow(x,2)-x)
        x = x + r3*(torch.pow(x,2)-x)
        enhance_image_1 = x + r4*(torch.pow(x,2)-x)     
        x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)     
        x = x + r6*(torch.pow(x,2)-x)   
        x = x + r7*(torch.pow(x,2)-x)
        x=x*0.6
        enhance_image = x + r8*(torch.pow(x,2)-x)
        r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)

        return enhance_image_1,enhance_image,r
```
经过改进得到如下结果：<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/17.png)
可以看出，输入图像经过低光照增强后能更好的显示出图像亮度和对比度的变化，能够恢复图像中的细微纹理，使得HDR重建图像效果更好。<br>

## 结果：


## 总结和讨论：
#### 邓棋丹：

&emsp;&emsp;我负责增加SRCNN技术提升图像的细节，使得重建后的HDR图像在细节表现上更加丰富和真实。SRCNN可以更好地保留亮部和暗部的细节，将低分辨率的LDR图像进行放大，提供更多的细节和更高的分辨率，有助于生成高质量的HDR图像。但是添加了低光照增强技术、去噪技术和SRCNN技术后，模型复杂性增加，训练时间可能会显著延长，需要更多的训练数据和更长的训练周期。<br>
​&emsp;&emsp;在收集资料时，我发现GAN的生成模型进行HDR重建是一个好方法，但它仍有自己的优势与劣势：<br>
优势：1.能够从大量的数据中学习并推广到新的场景，这使得它在不同的HDR重建任务中具有更广泛的适用性和泛化能力。2.对于受压缩伪影影响的图像的重建的问题，GAN可以通过学习和填充改善，提高重建图像的完整性和一致性。3.GAN能够捕获并增强图像中的细节，特别是在高动态范围场景下的细微结构和纹理。劣势：训练复杂的GAN模型通常需要大量的时间和计算资源，特别是对于需要高分辨率图像或复杂数据分布的任务，这是一个严重的限制。<br>

#### 范晓颖：

​&emsp;&emsp;我主要负责对原有代码进行改进以及增加低光照增强技术改善输入图像的光照条件，以便更好的进行hdr重建。在此次实验中，我们基于已有的从低动态范围(LDR)输入图像重建高动态范围(HDR)图像的方法基础上进行了组合式创新，添加了低光照增强、高分辨率、去噪等功能，将多种图像处理技术整合到一个统一的框架中，不仅提高了图像的视觉质量，还增强了模型的适用性和灵活性。<br>
在实验过程中，我发现可以进一步改进模型以提高模型在现实生活中解决实际问题的能力，我搜集资料得到了几种可行的方案：a.可以利用Transformer模型的注意力机制改进HDR重建模型的特征提取和处理能力;b.可以使用DenseNet的密集连接机制改进HDR重建模型的参数效率和梯度流动；c.结合GANs，通过生成对抗训练提高HDR图像的质量；d.可以结合MobileNets和SqueezeNet设计高效的HDR重建模型，使其适合移动设备和实时应用。在之后的日子里我会继续完善和改进这个实验，希望能达到满意的效果。

## 个人贡献声明：
邓棋丹：<br>
范晓颖：<br>
李奇静：<br>
蒯奇：<br>

## 引用参考：
>[1] Dong C , Loy C C , He K ,et al.Learning a Deep Convolutional Network for Image Super-Resolution[C]//ECCV.Springer International Publishing, 2014.DOI:10.1007/978-3-319-10593-2_13.
>[2] Dong C , Loy C C , Tang X .Accelerating the Super-Resolution Convolutional Neural Network[J].Springer, Cham, 2016.DOI:10.1007/978-3-319-46475-6_25.
>[3] Kim J , Lee J K , Lee K M .Accurate Image Super-Resolution Using Very Deep Convolutional Networks[J].IEEE, 2016.DOI:10.1109/CVPR.2016.182.
>[4] Lim B, Son S, Kim H, et al. Enhanced deep residual networks for single image super-resolution[C]//Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2017: 136-144.
>[5] Ledig C, Theis L, Huszár F, et al. Photo-realistic single image super-resolution using a generative adversarial network[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 4681-4690.
>[6] C. Guo et al., "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement," 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, 2020, pp. 1777-1786, doi: 10.1109/CVPR42600.2020.00185.
>[7] Gabriel Eilertsen, Joel Kronander, Gyorgy Denes, Rafał K. Mantiuk, and Jonas Unger. 2017. HDR image reconstruction from a single exposure using deep CNNs. ACM Trans. Graph. 36, 6, Article 178 (December 2017), 15 pages.
>[8] Nima Khademi Kalantari and Ravi Ramamoorthi. 2017. Deep high dynamic range imaging of dynamic scenes. ACM Trans. Graph. 36, 4, Article 144 (August 2017), 12 pages. 
>[9] Zhang K , Zuo W , Chen Y ,et al.Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising[J].IEEE Transactions on Image Processing, 2016, 26(7):3142-3155.DOI:10.1109/TIP.2017.2662206.
