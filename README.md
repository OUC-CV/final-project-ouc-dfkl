# 项目报告

#### 视频网址：[https://www.baidu.com/](https://www.baidu.com/)

## 绪论：

&emsp;&emsp;高动态范围成像（简称HDR），是提供比普通图像更多图像细节和动态范围的技术。HDR通过扩展图像的亮度和色彩范围呈现更多亮部与暗部的细节，还原更真实的视觉效果。<br>
&emsp;&emsp;本文基于《HDR image reconstruction from a single exposure using deep CNNs》进行组合创新。参考论文提出了基于全卷积神经网络（CNN）从低动态范围（LDR）输入图像中重建HDR图像。这个方法的形式为混合动态范围自编码器，通过编码器网络对LDR输入图像进行编码，然后将编码后的图像输入到HDR解码器网络，进行HDR图像的重建。同时，网络还配备了跳跃连接使得在重建时更好地利用高分辨率图像细节。参考论文为了训练CNN模型，通过收集大量HDR图像数据集来创建训练集，然后对于每个HDR图像使用虚拟相机模型模拟一组对应的LDR曝光图像。接着最小化自定义HDR损失函数来优化网络权重，并为了进一步提高鲁棒性，利用迁移学习在从MIT Places数据库子集创建的大量模拟HDR图像上预训练权重。该方法解决了在饱和图像区域中预测丢失信息的问题，以实现从单次曝光重建HDR。<br>
​&emsp;&emsp;本文基于参考论文的复现代码增加了低光照增强技术，去噪技术和SRCNN超分辨率技术进行组合创新。我们增加低光照增强技术以改善输入图像的光照条件，这种增强技术通过提升图像中低亮度区域的细节，使得HDR重建在这些区域的效果更加出色。我们使用去噪技术来降低输入图像中的噪声，从而提升HDR图像的清晰度和质量。此外，我们还使用SRCNN超分辨率技术来提升图像的细节，使得重建后的HDR图像在细节表现上更加丰富和真实。<br>

## 相关工作：
&emsp;&emsp;深度卷积神经网络是一种深度学习架构，它通过多层卷积和池化操作有效地提取图像特征，用于图像分类、检测和重建等任务，以其在视觉识别和图像处理方面的卓越性能而成为该领域的核心技术。深度卷积神经网络在图像处理领域取得了显著的技术进步，特别是在从单次曝光的低动态范围图像中重建高动态范围图像方面，通过采用混合动态范围自编码器和跳跃连接等创新结构，显著提升了图像细节和色彩的恢复质量，实现了对复杂场景中高光和暗部区域的有效重建。然而，尽管这些技术在提高HDR图像重建的分辨率和视觉效果方面取得了突破，仍然面临着一些挑战，包括对极端亮度区域的准确恢复、压缩伪影的处理、量化噪声的抑制，以及在不同光照和场景条件下保持算法鲁棒性的问题。此外，如何进一步提升网络对各种相机响应和图像压缩算法的适应性，以及如何优化网络结构以处理更高分辨率的图像，也是当前研究中需要解决的关键问题。<br>
​&emsp;&emsp;SRCNN是超分辨率任务中最早且最具影响力的深度学习方法之一,由Dong等人提出[1]。超分辨率就是把低分辨率（LP）图像放大为高分辨率（HP）图像。SRCNN通过一个简单的三层卷积网络实现了这一功能。SRCNN的工作流程包括三步：首先，输入LP图像，经过双三次（bicubic）插值，被放大成目标尺寸，得到Y；然后，通过三层卷积网络拟合非线性映射；最后，输出HR图像结果F(Y)。<br>
​&emsp;&emsp;此后的研究工作在SRCNN的基础上进行了改进。FSRCNN[2]通过引入更深层次的网络结构和去除冗余计算，进一步提升了超分辨率性能和计算效率。VDSR[3]通过增加网络深度和引入残差学习，显著提高了超分辨率效果。EDSR[4]则通过移除批归一化层和进一步增加网络深度，达到了超分辨率任务的新高度。然而，SRCNN及其后续方法仍存在一些局限性。它们在处理极高放大倍数或图像内容复杂度较高的场景时，可能会出现伪影或细节丢失的问题。近年来，生成对抗网络（GAN）和其他深度学习方法在超分辨率任务中的应用也展示了更好的视觉效果和性能[5]。<br>
​&emsp;&emsp;Zero-Reference Deep Curve Estimation (Zero-DCE) 是一种用于低光照图像增强的基于卷积神经网络（CNN）架构的方法。该方法通过无监督的方式估计一组光线增强曲线，而不依赖任何参考图像。这种网络设计的目的是通过调整像素值来提高低光照条件下拍摄图像的可见度。低光照增强技术在深度学习的推动下取得了显著进展，但在真实数据的获取、计算资源的优化、时间一致性的处理以及模型效率方面仍有很多亟待解决的问题，这些问题的解决将进一步推动该领域的发展和应用。<br>
&emsp;&emsp;残差学习是一种深度学习技术，它通过训练网络学习输入和输出之间的残差，而不是直接学习映射关系。在图像去噪的上下文中，残差学习允许深度卷积神经网络专注于学习噪声图像与干净图像之间的差异，而不是干净图像本身。残差学习技术的最新进展已经推动了深度学习在多个视觉任务中的性能，特别是在图像去噪领域，通过训练网络专注于预测噪声与干净图像之间的残差，这种方法能够显著提升去噪效果并加速训练过程；然而，尽管取得了这些进步，残差学习仍然面临着一些挑战，包括对大量训练数据的依赖、在处理极端噪声水平或复杂图像结构时的性能局限，以及在某些情况下可能需要进一步优化以提高模型的泛化能力和鲁棒性。未来的研究需要探索更高效的网络架构、损失函数和正则化技术，以解决这些问题并进一步提升残差学习在实际应用中的效能。<br>

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
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/4.png)<br>

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
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/5.png)<br>
从YCbCr到RGB公式：<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/6.png)<br>
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
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/7.png)<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/8.png)<br>


接着，想在经过低光照增强的图上使用超分辨率技术。发现ldr_enhance是一个 NumPy 数组和super_resolution函数输入图像类型不一致，要将其先转换为PIL图像对象，并且确保 ldr_enhance中的值在[0, 1]范围内。如果超出此范围，需要在传递给PIL前进行归一化处理。<br>
```python
image = pil_image.fromarray((ldr_enhance * 255).astype(np.uint8))
```

在ldr_enhance图像上使用超分辨率后进行HDR重建：<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/9.png)<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/10.png)<br>


#### 范晓颖：
我们基于已有的从低动态范围(LDR)输入图像重建高动态范围(HDR)图像的方法基础上进行了组合式创新，我主要负责对原有代码进行改进以及增加低光照增强技术改善输入图像的光照条件，以便更好的进行hdr重建。<br>
在最开始我运行原有代码时发现运行效果不是很好，重建后的图像与输入图像差别不大，多次修改阈值后发现当tau参数较大时，重建图像与输入图像相同；tau参数较小时，重建图像呈现紫色。<br>
tau=0.95时：<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/11.png)<br>
tau=0.05:<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/12.png)<br>
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
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/13.png)<br>
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
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/14.png)<br>
原有代码经过改进后得到的训练和验证损失函数图像如下：<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/15.png)<br>
为了改善视觉质量，更好地反映真实世界场景的光照变化，生成高质量HDR图像，我增加了低光照增强功能。该功能的添加我参考了论文《Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement》，将该论文提出的低光照增强方法组合到我们原有代码上，得到如下结果：<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/16.png)<br>
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
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/17.png)<br>
可以看出，输入图像经过低光照增强后能更好的显示出图像亮度和对比度的变化，能够恢复图像中的细微纹理，使得HDR重建图像效果更好。<br>

#### 李奇静$蒯奇：
首先，我们尝试在原本的数据集上复现代码<br>
论文的代码先定义了一个名为 DnCNN 的深度学习卷积神经网络模型类，用于本文的图像去噪任务。通过卷积神经网络（CNN）来学习去噪图像的残差。
第一层是一个卷积层，将输入通道数转换为特征图数量。<br>
紧接着是一个 ReLU 激活函数，增加网络的非线性能力。<br>
中间的层是多个卷积层、批归一化层和 ReLU 激活函数的组合。<br>
最后一层是一个卷积层，将特征图数量转换回输入的通道数。<br>
nn.Sequential(*layers) 将所有层组合成一个顺序容器，方便前向传播。<br>
模型输入是带噪声的图像，输出是预测的噪声，通过预测噪声的大小然后删去，可以得到去噪后的图像。<br>
我们所用到的深度卷积神经网络主要由多个卷积层、激活函数（ReLU）和批量归一化层组成。这些层级组合起来形成一个深度网络，可以处理复杂的图像去噪任务。<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/20.png)<br>
然后对模型进行训练，主要是基于残差学习的思想。具体来说，就是通过学习带噪声图像和干净图像之间的差异（噪声），来去除图像中的噪声。<br>
其中残差学习的核心思想是让网络学习输入图像与干净图像之间的差异（即噪声）。假设我们有一个带噪声的输入图像y，干净图像x，以及噪声v满足y=x+v。目标是训练一个网络F(y)来预测残差v，然后通过x=y−F(y)得到去噪后的图像。<br>
具体训练时，先创建我们刚刚建立的DnCNN 模型。然后使用 Kaiming 初始化方法初始化模型的权重。设置均方误差为损失函数，用于衡量模型输出和目标值之间的误差。（也就是学习噪声的残差学习的思想）并且使用 Adam 优化器进行模型参数的优化，指定初始学习率。<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/21.png)<br>
然后通过遍历数据集，先为数据添加噪声，然后进行前向传播和损失计算，根据这些进行学习率的调整，以及模型性能的评估等操作。<br>
其中损失函数的计算大致如下：假设我们有一个训练集 ，其中yi是带噪声的图像，xi是对应的干净图像。我们要最小化以下均方误差（MSE）损失函数：<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/22.png)<br>
其中，θ是网络参数， F(yi;θ)是预测的噪声残差。<br>
最后我们用训练好的模型在原本的数据集上进行预测，同样的步骤加载刚刚训练好的模型权重运用在加载的数据集上，通过遍历每个测试图像文件：先对图片添加噪声（因为数据集中是没有噪声的图片，通过加入确定大小的噪声，既能起到模型训练的效果又能判断模型预测噪声水平是否正确）然后计算去噪后的图像，计算并打印每个图像的PSNR。至于这个PSNR，就是评估我们模型训练效果的评价指标。<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/23.png)<br>
通过完整的第一个步骤我们得到了这样的结果<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/24.png)<br>
在原论文的复现中我们成功还原了应用了去噪网络模型后的效果图像，可以看到明显的去噪效果，虽然存在颜色失真和细节损失。<br>
但是当我们开始组合到我们选择了第一篇论文（后称原论文）的方法中时，我们遇到了问题，我们首先尝试了将第一个步骤中的网络模型预测部分的代码封装成一个函数denoise_image（）放到我们新的predict（组合各个方法的总文件）文件中去，如下图<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/25.png)<br>
但是总是出现通道数不匹配的问题，经过很久的修改强行让通道数匹配之后运行出来的结果也十分不理想<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/26.png)<br>
经过一系列的修改，改了很久，收效甚微，仅有一次改成了灰色绿色的图像（当时就直接继续改了，没有存下来截图），只有线条清晰（像是黑色描边的那种图像灰色图像）所以也不能算好结果。后面越修改越不对越混乱，最后连这个图像也恢复不回来了。<br>
后来我们决定重头再来一次，也就有了先复现论文再进行实验的步骤，虽然比第一次进行的顺利，但是最终的结果还是不尽人意，大致如图。（因为最终的结果不好，所以说再多我们很努力的进行了修改也是无用功，结果不好就是没有完成工作，所以后面我们也为此进行了深刻的思考）<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/27.png)<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/28.png)<br>


 
## 结果：
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/18.png)<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/19.png)<br>
该结果图第一行是输入的ldr图像、重建后的图像以及理想图。<br>
第二行是使用低光照增强技术后的ldr图像、重建后的图像以及理想图。<br>
第三行是使用去噪技术后的ldr图像、重建后的图像以及理想图。<br>
第四行是使用SRCNN技术后的ldr图像、重建后的图像以及理想图。<br>
第五行是对使用低光照增强技术后的ldr图像使用SRCNN技术后的图像、重建后的图像以及理想图。<br>
<br>
数据集的量化信息：<br>
数据集划分比例：80%训练集和20%测试集<br>
样本数量：1110个样本<br>
特征数量：图像的尺寸为256*256和颜色通道数为3个通道<br>
训练时间：训练时间大约6个小时<br>




## 总结和讨论：
#### 邓棋丹：

&emsp;&emsp;我负责增加SRCNN技术提升图像的细节，使得重建后的HDR图像在细节表现上更加丰富和真实。SRCNN可以更好地保留亮部和暗部的细节，将低分辨率的LDR图像进行放大，提供更多的细节和更高的分辨率，有助于生成高质量的HDR图像。但是添加了低光照增强技术、去噪技术和SRCNN技术后，模型复杂性增加，训练时间可能会显著延长，需要更多的训练数据和更长的训练周期。<br>
​&emsp;&emsp;在收集资料时，我发现GAN的生成模型进行HDR重建是一个好方法，但它仍有自己的优势与劣势：<br>
优势：1.能够从大量的数据中学习并推广到新的场景，这使得它在不同的HDR重建任务中具有更广泛的适用性和泛化能力。2.对于受压缩伪影影响的图像的重建的问题，GAN可以通过学习和填充改善，提高重建图像的完整性和一致性。3.GAN能够捕获并增强图像中的细节，特别是在高动态范围场景下的细微结构和纹理。劣势：训练复杂的GAN模型通常需要大量的时间和计算资源，特别是对于需要高分辨率图像或复杂数据分布的任务，这是一个严重的限制。<br>

#### 范晓颖：

​&emsp;&emsp;我主要负责对原有代码进行改进以及增加低光照增强技术改善输入图像的光照条件，以便更好的进行hdr重建。在此次实验中，我们基于已有的从低动态范围(LDR)输入图像重建高动态范围(HDR)图像的方法基础上进行了组合式创新，添加了低光照增强、高分辨率、去噪等功能，将多种图像处理技术整合到一个统一的框架中，不仅提高了图像的视觉质量，还增强了模型的适用性和灵活性。<br>
在实验过程中，我发现可以进一步改进模型以提高模型在现实生活中解决实际问题的能力，我搜集资料得到了几种可行的方案：a.可以利用Transformer模型的注意力机制改进HDR重建模型的特征提取和处理能力;b.可以使用DenseNet的密集连接机制改进HDR重建模型的参数效率和梯度流动；c.结合GANs，通过生成对抗训练提高HDR图像的质量；d.可以结合MobileNets和SqueezeNet设计高效的HDR重建模型，使其适合移动设备和实时应用。在之后的日子里我会继续完善和改进这个实验，希望能达到满意的效果。<br>

#### 李奇静$蒯奇：

1.Denoise本身: 其实在原论文的恢复中上应用了去噪网络后的图像，可以看到明显的去噪效果，所以论文本身的方法也一定是可行的。<br>
![](https://github.com/OUC-CV/final-project-ouc-dfkl/blob/main/image/29.png)<br>
我们的去噪部分没有完成的很好，导致最终的结果无法用我们的方法，我们感到很愧疚也很遗憾，但是既然现在结束了，我们就需要回过头来思考，出现问题的根本原因就在于，我们上来就一股脑的开始，不去考虑论文是否可行、两个论文的方法是否适合是否可以融合，而是不管不顾直接开始添加、改错，到最后发现效果不理想的时候，还是一股脑的修改程序，单纯的询问为什么效果不够好，（仅仅给出了一些类似于“可能为LDR图像的动态范围较低，而HDR图像包含更多的亮度和细节信息。直接应用LDR去噪网络可能忽略了这种动态范围的差异，导致去噪后图像失真等等”这种不痛不痒的说法）却还是不知道思考，一个劲的修改代码，做无用功；虽然在最后明白了一点冷静了一段时间再次尝试，先恢复论文原本的方法，发现确实可行，然后再重新添加；但是这个时候还是没有考虑两个模型是否合适的问题，再次尝试失败后才发现，这根本就不是合适的模型，仔细思考过后我们才发现，我们找到的去噪方法的论文是比较适用于有噪音的图片的基础上进行去噪，并且只适用于单通道灰色图像，这样训练出来的模型当然是不可以被三通道rgb图像使用的，但是其实后来还有最后一点时间的时候我有再尝试过一次，想着把这个模型单通道的部分改成可以处理彩色图像的方法，比如<br>
（1）修改我的DNN模型，将其变成3通道，net = DnCNN(channels=3, num_of_layers=opt.num_of_layers)<br>
（2）在读入数据时保留3通道Img = cv2.imread(f) <br>
Img = normalize(np.float32(Img)) <br>
Img = np.transpose(Img, (2, 0, 1)) <br>
Img = np.expand_dims(Img, 0) 等等<br>
但是最终由于时间原因以及能力水平有限，还是没能成功，很遗憾。但是我认为这是可以修改的方向，因为我觉得颜色的问题，主要还是由模型处理的通道数的限制造成的。<br>
2.De_Reconstruction: 去噪后重建的图像，显示了去噪后的HDR效果，虽然去噪效果不理想，但还是能明显看出是有效果的，总之！虽然效果不好，但是我们还是成功的将两个方法融合到一起了。<br>
总之，通过这次的项目，我学到了特别特别多，不仅仅是知识，还有一些做项目时的团队配合沟通、通宵做项目的神奇经历、以及导致自己任务失败的根本原因，都是我这次项目获得的宝贵的东西：<br>
1.在开始实现论文组合式创新或者是一些其他创新工作之前，要保持冷静不能急躁，深刻掌握各个方法，理解理论基础，也就是知己知彼，仔细评估所选择的方法是否适用，理解每个方法的优点和局限性，这是非常重要的。直接开始实现和调试，虽然可以快速上手，但可能会忽略方法本身的适用性和局限性。应该先进行理解和分析，以确定方法是否适合当前任务。<br>

2.在正式尝试融合不同方法时，需要细致的规划和设计。这包括确定每个方法的输入输出、融合点以及可能的冲突和解决方案。没有规划和设计就开始实施，可能会导致重复劳动和低效工作。比如我们就来回做了三次才发现原来从一开始就是错的。<br>

3.在项目进展不顺利时，应该冷静下来，反思当前策略，并进行必要的调整。可以与团队成员进行沟通交换想法获得灵感。<br>

4.也是最根本的提升技术能力：解决复杂问题从最源头上说还是需要自身实力过硬，而且仅仅依靠代码实现，而不理解背后的理论，可能会在遇到问题时束手无策。应该结合理论学习（阅读文献）和实践学习（参加培训和实践项目），逐步提升自己的能力。<br>




## 个人贡献声明：
邓棋丹：<br>
范晓颖：<br>
李奇静：<br>
蒯奇：<br>

## 引用参考：
>[1] Dong C , Loy C C , He K ,et al.Learning a Deep Convolutional Network for Image Super-Resolution[C]//ECCV.Springer International Publishing, 2014.DOI:10.1007/978-3-319-10593-2_13.<br>
>[2] Dong C , Loy C C , Tang X .Accelerating the Super-Resolution Convolutional Neural Network[J].Springer, Cham, 2016.DOI:10.1007/978-3-319-46475-6_25.<br>
>[3] Kim J , Lee J K , Lee K M .Accurate Image Super-Resolution Using Very Deep Convolutional Networks[J].IEEE, 2016.DOI:10.1109/CVPR.2016.182.<br>
>[4] Lim B, Son S, Kim H, et al. Enhanced deep residual networks for single image super-resolution[C]//Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2017: 136-144.<br>
>[5] Ledig C, Theis L, Huszár F, et al. Photo-realistic single image super-resolution using a generative adversarial network[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 4681-4690.<br>
>[6] C. Guo et al., "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement," 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, 2020, pp. 1777-1786, doi: 10.1109/CVPR42600.2020.00185.<br>
>[7] Gabriel Eilertsen, Joel Kronander, Gyorgy Denes, Rafał K. Mantiuk, and Jonas Unger. 2017. HDR image reconstruction from a single exposure using deep CNNs. ACM Trans. Graph. 36, 6, Article 178 (December 2017), 15 pages.<br>
>[8] Nima Khademi Kalantari and Ravi Ramamoorthi. 2017. Deep high dynamic range imaging of dynamic scenes. ACM Trans. Graph. 36, 4, Article 144 (August 2017), 12 pages. <br>
>[9] Zhang K , Zuo W , Chen Y ,et al.Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising[J].IEEE Transactions on Image Processing, 2016, 26(7):3142-3155.DOI:10.1109/TIP.2017.2662206.<br>
