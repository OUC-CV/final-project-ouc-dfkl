import torch
import cv2
from lowlight_test import lowlight
import model as mod
from model import DnCNN, Model
import matplotlib.pyplot as plt
import numpy as np
import time

def color_balance(image):
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab_image) 
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) 
    L_eq = clahe.apply(L)  
    lab_image_eq = cv2.merge((L_eq, A, B))  
    balanced_image = cv2.cvtColor(lab_image_eq, cv2.COLOR_LAB2RGB)  

    balanced_image = balanced_image / 255.0
    return balanced_image

def itm(model, input_image, tau):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    # input_image = input_image / 255.0
    # 调用颜色通道均衡处理函数
    input_image = color_balance(input_image)

    input_image = torch.tensor(input_image).permute(2, 0, 1).unsqueeze(0).float()
    input_image = input_image.to(device)
    
    max_c = input_image[0].max(dim=0).values - tau
    max_c[max_c < 0] = 0
    alpha = max_c / (1 - tau)
    
    with torch.no_grad():
        output_image = model(input_image)
    
    output_image = ((1 - alpha) * (input_image ** 2) + alpha * output_image).squeeze().permute(1, 2, 0).cpu().detach().numpy()
    output_image = np.clip(output_image, 0, 1)

    return output_image



def lowlight(input_image):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    data_lowlight = input_image/255.0
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.to(device)
    data_lowlight = data_lowlight.permute(2,0,1)
    data_lowlight = data_lowlight.unsqueeze(0)

    DCE_net = mod.enhance_net_nopool().to(device)
    DCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth', map_location=device))
    start = time.time()
    _,enhanced_image,_ = DCE_net(data_lowlight)

    end_time = (time.time() - start)
    print(end_time)
    image_np = enhanced_image.numpy() if enhanced_image.ndim == 3 else enhanced_image[0].detach().numpy()

# 如果图像是多通道的，我们需要从 (C, H, W) 转换到 (H, W, C) 格式
    if image_np.ndim == 3 and image_np.shape[0] in {1, 3}:  # 1 表示灰度图，3 表示彩色图
        image_np = image_np.transpose(1, 2, 0)
    image_np = np.clip(image_np, 0, 1) 
    print("lowlight success!")
    return image_np

# def denoise_image(input_image, model, tau):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     # 调用颜色通道均衡处理函数，这里假设 color_balance 函数已经定义
#     input_image = color_balance(input_image)

#     # 数据预处理
#     input_image = input_image / 255.0  # 归一化到 [0, 1] 范围
#     input_image = np.transpose(input_image, (2, 0, 1))  # 转换为 PyTorch 期望的通道顺序 (C, H, W)
#     input_image = np.expand_dims(input_image, axis=0)  # 添加批处理维度 (1, C, H, W)
#     input_image = torch.tensor(input_image).float().to(device)

#     # 计算 max_c 和 alpha
#     max_c = torch.max(input_image[0], dim=0).values - tau
#     max_c[max_c < 0] = 0
#     alpha = max_c / (1 - tau)

#     # 模型推断
#     with torch.no_grad():
#         output_image = model(input_image)

#     # 后处理
#     output_image = (1 - alpha) * (input_image ** 2) + alpha * output_image
#     output_image = torch.squeeze(output_image).permute(1, 2, 0).cpu().numpy()
#     output_image = np.clip(output_image, 0, 1)  # 确保像素值在 [0, 1] 范围内

#     return output_image

from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio
# 定义去噪函数
def denoise_image(input_image, model, noise_level):
    def normalize(data):
        return data / 255.0
    
    def batch_PSNR(img, imclean, data_range):
        Img = img.cpu().numpy().astype(np.float32)
        Iclean = imclean.cpu().numpy().astype(np.float32)
        PSNR = 0
        for i in range(Img.shape[0]):
            PSNR += peak_signal_noise_ratio(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
        return PSNR / Img.shape[0]

    Img = normalize(np.float32(input_image[:, :, 0]))
    Img = np.expand_dims(Img, 0)
    Img = np.expand_dims(Img, 1)
    ISource = torch.Tensor(Img)

    noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=noise_level / 255.0)
    INoisy = ISource + noise
    ISource, INoisy = Variable(ISource), Variable(INoisy)

    with torch.no_grad():
        Out = torch.clamp(INoisy - model(INoisy), 0.0, 1.0)
    
    psnr = batch_PSNR(Out, ISource, 1.0)
    #print("PSNR of denoised image: %f" % psnr)

    denoised_img = Out.squeeze().cpu().numpy()
    denoised_img = (denoised_img * 255).astype(np.uint8)
    print("Denoise success!")
    return denoised_img




#超分辨率
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr
import torch.backends.cudnn as cudnn
from model import SRCNN
import PIL.Image as pil_image
import os

def super_resolution(image, weights_file='./weight/srcnn_x3.pth', scale=3):
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    if isinstance(image, np.ndarray):
        image = pil_image.fromarray(image)
    
    image_width = (image.width // scale) * scale  
    image_height = (image.height // scale) * scale  
    
    image_resized = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    image_resized = image_resized.resize((image_resized.width // scale, image_resized.height // scale), resample=pil_image.BICUBIC)
    image_resized = image_resized.resize((image_resized.width * scale, image_resized.height * scale), resample=pil_image.BICUBIC)
    
    save_path_bicubic = os.path.join('test', f'bicubic_x{scale}.png')
    image_resized.save(save_path_bicubic)

    image_np = np.array(image_resized).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image_np)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)
    #print(f"Shape of y before model prediction: {y.shape}")

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    #print(f"Shape of preds after model prediction: {preds.shape}")

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)

    save_path_srcnn = os.path.join('test', f'srcnn_x{scale}.png')
    output_image = pil_image.fromarray(output)
    output_image.save(save_path_srcnn)

    output_np_normalized = output / 255.0

    print("super_resolution success!")

    return output_np_normalized

if __name__ == '__main__':
    # Load model
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = Model(device).to(device)
    model.eval()
    model.load_state_dict(torch.load('output\\weight.pth', map_location=device))

    # 加载图片
    ldr = cv2.imread('test\\1.png')
    ldr = cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB)
    #低光照
    ldr_enhance = lowlight(ldr)
    #降噪
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_denoise = DnCNN(channels=1, num_of_layers=17).to(device)
    state_dict = torch.load('logs/DnCNN-B/net.pth', map_location=device)
    new_state_dict = {}

    for k, v in state_dict.items():
    # 将 dncnn.0.weight 的形状从 [64, 64, 3, 3] 调整为 [1, 64, 3, 3]
     if k.startswith('dncnn.0'):
         v = v[:, :1, :, :]  # 调整通道数
     new_state_dict[k] = v

    # 加载修正后的状态字典
    model_denoise.load_state_dict(new_state_dict,strict=False)
    model_denoise.eval()

    ldr_denoised = denoise_image(ldr, model_denoise, noise_level=25.0) 
    #ldr_denoised = denoise_image(ldr, model_denoise, tau=0.75) 
    
    # print("ldr_enhance_denoise",ldr_denoised)
    # print(ldr_denoised.shape)  # 应该输出 (H, W, 3)
    ldr_denoised_rgb = cv2.cvtColor(ldr_denoised, cv2.COLOR_GRAY2RGB)
    ldr_denoised=ldr_denoised/255

    #超分辨率
    ldr_pil = pil_image.fromarray(ldr)
    ldr_super_resolution = super_resolution(ldr_pil, 'weight/srcnn_x3.pth')
    # print(f"Shape of ldr_super_resolution before model prediction: {ldr_super_resolution.shape}")

    #在enhance的基础上
    # image = pil_image.fromarray((ldr_enhance * 255).astype(np.uint8))
    # ldr_super_resolution = super_resolution(image, 'weight/srcnn_x3.pth')
    
    #在denoise的基础上
    #image = pil_image.fromarray((ldr_denoise * 255).astype(np.uint8))
    #ldr_super_resolution = super_resolution(image, 'weight/srcnn_x3.pth')


    #原图
    hdr = cv2.imread('test\\1.hdr', cv2.IMREAD_ANYDEPTH)
    hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)
    ldr = ldr / 255.0
    #原论文
    hdr_reconstructed = itm(model, ldr, tau=0.75)
    #增强后+原论文
    en_reconstructed = itm(model, ldr_enhance, tau=0.75)
    #去噪后原论文
    de_reconstructed = itm(model, ldr_denoised_rgb, tau=0.75)

    ldr_super_resolution_resized = cv2.resize(ldr_super_resolution, (1024, 768))
    super_reconstructed = itm(model, ldr_super_resolution_resized, tau=0.75)

    ####输出图像结果对比
    # Save result
    #fig = plt.figure()
    fig = plt.figure(figsize=(12, 10))  # 调整画布大小

    fig.add_subplot(4, 3, 1)
    plt.imshow(ldr)
    plt.axis('off')
    plt.title('Input')

    fig.add_subplot(4, 3, 2)
    plt.imshow(hdr_reconstructed ** (1 / 2))
    plt.axis('off')
    plt.title('Reconstruction')

    fig.add_subplot(4, 3, 3)
    plt.imshow(hdr ** (1 / 2))
    plt.axis('off')
    plt.title('Ground truth')

    fig.add_subplot(4, 3, 4)
    plt.imshow(ldr_enhance)
    plt.axis('off')
    plt.title('Enhance')

    fig.add_subplot(4, 3, 5)
    plt.imshow(en_reconstructed ** (1 / 2))
    plt.axis('off')
    plt.title('En_Reconstruction')

    fig.add_subplot(4, 3, 6)
    plt.imshow(hdr ** (1 / 2))
    plt.axis('off')
    plt.title('Ground truth')


    fig.add_subplot(4, 3, 7)
    plt.imshow(ldr_denoised)
    plt.axis('off')
    plt.title('Denoise')

    fig.add_subplot(4, 3, 8)
    plt.imshow(de_reconstructed ** (1 / 2.2))
    plt.axis('off')
    plt.title('De_Reconstruction')

    fig.add_subplot(4, 3, 9)
    plt.imshow(hdr ** (1 / 2.2))
    plt.axis('off')
    plt.title('Ground truth')

    fig.add_subplot(4, 3, 10)
    plt.imshow(ldr_super_resolution ** (1 / 2.2))
    plt.axis('off')
    plt.title('super_resolution')

    fig.add_subplot(4, 3, 11)
    plt.imshow(super_reconstructed ** (1 / 2.2))
    plt.axis('off')
    plt.title('Super_Reconstruction')

    fig.add_subplot(4, 3, 12)
    plt.imshow(hdr ** (1 / 2.2))
    plt.axis('off')
    plt.title('Ground truth')

    plt.savefig('test\\results\\1.png', dpi=300, bbox_inches='tight')
    plt.show()
