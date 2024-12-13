import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import DnCNN
from utils import *
from skimage.metrics import peak_signal_noise_ratio

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs/DnCNN-B/", help='path of log files')#不知道噪声水平
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
parser.add_argument("--output_dir", type=str, default="output/kq", help='directory to save denoised images')
opt = parser.parse_args()

def normalize(data):
    return data / 255.

def batch_PSNR(img, imclean, data_range):
    Img = img.cpu().numpy().astype(np.float32)
    Iclean = imclean.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return PSNR / Img.shape[0]

def main():
    # Build model
    print('Loading model ...\n')
    net = DnCNN(channels=3, num_of_layers=opt.num_of_layers)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cpu()  # Change to CPU
    model_path = os.path.join(opt.logdir, 'net.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist. Please place the model file at this location.")

    # Load the saved model weights
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Remove unexpected keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in model.state_dict() and model.state_dict()[k].size() == v.size():
            new_state_dict[k] = v
        else:
            print(f"Skipping {k} due to size mismatch or unexpected key")

    # Load the new state_dict
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    # Load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('datanoise', opt.test_data, '*.png'))
    files_source.sort()
    
    # Create output directory if it does not exist
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    
    # Process data
    psnr_test = 0
    for f in files_source:
        # Image
        Img = cv2.imread(f)
        Img = normalize(np.float32(Img[:, :, 0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        
        # Noise
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / 255.)
        
        # Noisy image
        INoisy = ISource + noise
        ISource, INoisy = Variable(ISource), Variable(INoisy)
        
        with torch.no_grad(): # this can save much memory
            Out = torch.clamp(INoisy - model(INoisy), 0., 1.)
        
        # Calculate PSNR
        psnr = batch_PSNR(Out, ISource, 1.)
        psnr_test += psnr
        print("%s PSNR %f" % (f, psnr))
        
        # Save the denoised image
        denoised_img = Out.squeeze().cpu().numpy()
        denoised_img = (denoised_img * 255).astype(np.uint8)
        output_path = os.path.join(opt.output_dir, os.path.basename(f))
        cv2.imwrite(output_path, denoised_img)
    
    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__":
    main()

# import cv2
# import os
# import argparse
# import glob
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# from model import DnCNN
# from utils import *
# from skimage.metrics import peak_signal_noise_ratio

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parser = argparse.ArgumentParser(description="DnCNN_Test")
# parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
# parser.add_argument("--logdir", type=str, default="logs/DnCNN-B/", help='path of log files')#不知道噪声水平
# parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
# parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
# opt = parser.parse_args()

# def normalize(data):
#     return data / 255.

# def batch_PSNR(img, imclean, data_range):
#     Img = img.cpu().numpy().astype(np.float32)
#     Iclean = imclean.cpu().numpy().astype(np.float32)
#     PSNR = 0
#     for i in range(Img.shape[0]):
#         PSNR += peak_signal_noise_ratio(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
#     return PSNR / Img.shape[0]

# def main():
#     # Build model
#     print('Loading model ...\n')
#     net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
#     device_ids = [0]
#     model = nn.DataParallel(net, device_ids=device_ids).cpu()  # Change to CPU
#     model_path = os.path.join(opt.logdir, 'net.pth')
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file {model_path} does not exist. Please place the model file at this location.")

#     # Load the saved model weights
#     state_dict = torch.load(model_path, map_location=torch.device('cpu'))

#     # Remove unexpected keys
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         if k in model.state_dict() and model.state_dict()[k].size() == v.size():
#             new_state_dict[k] = v
#         else:
#             print(f"Skipping {k} due to size mismatch or unexpected key")

#     # Load the new state_dict
#     model.load_state_dict(new_state_dict, strict=False)
#     model.eval()
    
#     # Load data info
#     print('Loading data info ...\n')
#     files_source = glob.glob(os.path.join('datanoise', opt.test_data, '*.png'))
#     files_source.sort()
    
#     # Process data
#     psnr_test = 0
#     for f in files_source:
#         # Image
#         Img = cv2.imread(f)
#         Img = normalize(np.float32(Img[:, :, 0]))
#         Img = np.expand_dims(Img, 0)
#         Img = np.expand_dims(Img, 1)
#         ISource = torch.Tensor(Img)
        
#         # Noise
#         noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / 255.)
        
#         # Noisy image
#         INoisy = ISource + noise
#         ISource, INoisy = Variable(ISource), Variable(INoisy)
        
#         with torch.no_grad(): # this can save much memory
#             Out = torch.clamp(INoisy - model(INoisy), 0., 1.)
        
#         psnr = batch_PSNR(Out, ISource, 1.)
#         psnr_test += psnr
#         print("%s PSNR %f" % (f, psnr))
    
#     psnr_test /= len(files_source)
#     print("\nPSNR on test data %f" % psnr_test)

# if __name__ == "__main__":
#     main()























# import cv2
# import os
# import argparse
# import glob
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# from model import DnCNN
# from utils import *
# from skimage.metrics import peak_signal_noise_ratio

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parser = argparse.ArgumentParser(description="DnCNN_Test")
# parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
# parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
# parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
# parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
# opt = parser.parse_args()

# def normalize(data):
#     return data / 255.

# def batch_PSNR(img, imclean, data_range):
#     Img = img.cpu().numpy().astype(np.float32)
#     Iclean = imclean.cpu().numpy().astype(np.float32)
#     PSNR = 0
#     for i in range(Img.shape[0]):
#         PSNR += peak_signal_noise_ratio(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
#     return PSNR / Img.shape[0]

# def main():
#     # Build model
#     print('Loading model ...\n')
#     net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
#     device_ids = [0]
#     model = nn.DataParallel(net, device_ids=device_ids).cpu()  # Change to CPU
#     model_path = os.path.join(opt.logdir, 'net.pth')
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file {model_path} does not exist. Please place the model file at this location.")
#     model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#     model.eval()
    
#     # Load data info
#     print('Loading data info ...\n')
#     files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
#     files_source.sort()
    
#     # Process data
#     psnr_test = 0
#     for f in files_source:
#         # Image
#         Img = cv2.imread(f)
#         Img = normalize(np.float32(Img[:, :, 0]))
#         Img = np.expand_dims(Img, 0)
#         Img = np.expand_dims(Img, 1)
#         ISource = torch.Tensor(Img)
        
#         # Noise
#         noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / 255.)
        
#         # Noisy image
#         INoisy = ISource + noise
#         ISource, INoisy = Variable(ISource), Variable(INoisy)
        
#         with torch.no_grad(): # this can save much memory
#             Out = torch.clamp(INoisy - model(INoisy), 0., 1.)
        
#         psnr = batch_PSNR(Out, ISource, 1.)
#         psnr_test += psnr
#         print("%s PSNR %f" % (f, psnr))
    
#     psnr_test /= len(files_source)
#     print("\nPSNR on test data %f" % psnr_test)

# if __name__ == "__main__":
#     main()


# import cv2
# import os
# import argparse
# import glob
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# from model import DnCNN
# from utils import *
# from skimage.metrics import peak_signal_noise_ratio

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parser = argparse.ArgumentParser(description="DnCNN_Test")
# parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
# parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
# parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
# parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
# opt = parser.parse_args()

# def normalize(data):
#     return data/255.

# def main():
#     # Build model
#     print('Loading model ...\n')
#     net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
#     device_ids = [0]
#     model = nn.DataParallel(net, device_ids=device_ids).cpu()
#     model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
#     model.eval()
#     # load data info
#     print('Loading data info ...\n')
#     files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
#     files_source.sort()
#     # process data
#     psnr_test = 0
#     for f in files_source:
#         # image
#         Img = cv2.imread(f)
#         Img = normalize(np.float32(Img[:,:,0]))
#         Img = np.expand_dims(Img, 0)
#         Img = np.expand_dims(Img, 1)
#         ISource = torch.Tensor(Img)
#         # noise
#         noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
#         # noisy image
#         INoisy = ISource + noise
#         ISource, INoisy = Variable(ISource), Variable(INoisy.cuda())
#         with torch.no_grad(): # this can save much memory
#             Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
#         ## if you are using older version of PyTorch, torch.no_grad() may not be supported
#         # ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
#         # Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
#         psnr = batch_PSNR(Out, ISource, 1.)
#         psnr_test += psnr
#         print("%s PSNR %f" % (f, psnr))
#     psnr_test /= len(files_source)
#     print("\nPSNR on test data %f" % psnr_test)

# if __name__ == "__main__":
#     main()