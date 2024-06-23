import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time


 
def lowlight(image_path):
	# os.environ['CUDA_VISIBLE_DEVICES']='0'
	device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

	#加载图像并预处理
	data_lowlight = Image.open(image_path)
	data_lowlight = (np.asarray(data_lowlight)/255.0)

	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.to(device)
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.unsqueeze(0)

	#加载预训练的深度学习模型
	DCE_net = model.enhance_net_nopool().to(device)
	DCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth', map_location=device))
	start = time.time()

	#将低光图像输入模型 DCE_net 中进行处理，得到增强后的图像 
	_,enhanced_image,_ = DCE_net(data_lowlight)

	end_time = (time.time() - start)
	print(end_time)
	image_path = image_path.replace('train','result')
	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

	torchvision.utils.save_image(enhanced_image, result_path)

if __name__ == '__main__':
# test_images
	with torch.no_grad():

		#遍历指定文件夹中的所有图像，并对每张图像应用 lowlight 函数进行低光照增强处理。
		filePath = 'data/train/'
		file_list = os.listdir(filePath)

		for file_name in file_list:
			test_list = glob.glob(filePath+file_name+"/*") 
			for image in test_list:
				# image = image
				print(image)
				lowlight(image)


                




		

