import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from skimage.measure import compare_ssim

# current compress rate value
global compr_rate

global original_image

global image_pack

global comp_rate_step

global full_filename

# Init global values 
image_pack = []
compr_rate = 5
comp_rate_step = 1
def psnr(mse):
	psnr_value = 20 * np.log10(255 / np.sqrt(mse))
	return psnr_value

def mse(compr_image):
	error = np.sum((original_image.astype('float') - compr_image.astype('float')) ** 2)
	error /= float(original_image.shape[0] * compr_image.shape[1])
	return error / 3

def ssim(compr_image):
	ssim_value = compare_ssim(original_image,
						compr_image,
						win_size=None,
						gradient=False,
						multichannel=True)
	return ssim_value

def compress(value):
	result, encimg = cv2.imencode('.jpg', original_image, [cv2.IMWRITE_JPEG_QUALITY, 100 - value])
	compr_image = cv2.imdecode(encimg, 1)
	return compr_image

def OnRateChange(event):
	# get current positions of four trackbars
	compr_rate = cv2.getTrackbarPos('compr_rate','new_image')
	new_image = compress(compr_rate)
	cv2.imshow("new_image", new_image)
	mse_value = mse(new_image)
	psnr_value = psnr(mse_value)
	ssim_value = ssim(new_image)
	print("Compression rate: " + str(compr_rate))
	print("MSE: " + str(mse_value))
	print("PSNR: " + str(psnr_value))
	print("SSIM: " + str(ssim_value))
	print("=========================================")


def prepareImagePack():
	for rate in range(0, 100, comp_rate_step):		
		new_filename = make_newfilename(rate)
		cv2.imwrite(new_filename, original_image, [cv2.IMWRITE_JPEG_QUALITY, 100 - rate])
	plot()

def make_newfilename(rate):
	splitted_filename = full_filename.split("\\");
	filename = splitted_filename.pop()
	path = "\\".join(splitted_filename)
	splitted_filename = filename.split(".")
	format = splitted_filename[1]
	filename = splitted_filename[0]
	new_filename = path + filename + "_" + str(rate) + "." + format
	return new_filename

def readImage(rate):
	image = cv2.imread(make_newfilename(rate))
	return image

def CalcMeasurePack(measure_name):
	ay = []
	h,w,c = original_image.shape
	cur_image = np.zeros((h, w, 3), dtype=np.uint8)
	if measure_name == "MSE":
		for x in range(0, 100, comp_rate_step):
			ay.append(mse(readImage(x)))
	elif measure_name == "PSNR":
		for x in range(0, 100, comp_rate_step):
			ay.append(psnr(mse(readImage(x))))
	else:
		for x in range(0, 100, comp_rate_step):
			ay.append(ssim(readImage(x)))
	return ay


def plot():
	ax = [x for x in range(0, 100, comp_rate_step)]
	ay = []
	names = ["MSE", "PSNR", "SSIM"]
	i = 0
	for name in names:
		ay = CalcMeasurePack(name)
		plt.figure(i+1)
		plt.plot(ax, ay)
		plt.title('{} Measure'.format(name))
		plt.xlabel('JPEG compression (%)')
		plt.ylabel('{} value'.format(name))
		i += 1
	for rate in range(0, 100, comp_rate_step):		
		new_filename = make_newfilename(rate)
		os.remove(new_filename)
	plt.show()

	

full_filename = askopenfilename();
original_image = cv2.imread(full_filename)

cv2.namedWindow('new_image')
x,y,c = original_image.shape
cv2.resizeWindow('new_image', x,y)
# create trackbars
cv2.createTrackbar('compr_rate','new_image',5,100,OnRateChange)
new_image = compress(compr_rate)
cv2.imshow('original_image',original_image)
cv2.imshow('original_image',new_image)
prepareImagePack()
cv2.waitKey()
cv2.destroyAllWindows()