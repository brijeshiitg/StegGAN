import os

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

from config import cfg
from utils.utils import threshold


class ToTensor(object):

	def __call__(self, sample):
		cover_image, message, cover_th, gt = sample['cover'], sample['msg'], sample['th'], sample['gt']
		message = message.astype(np.float32)
		message = torch.from_numpy(message)
		# print(message.size())
		message = torch.transpose(torch.transpose(message, 2, 0), 1, 2)
		message = message / 255.0

		cover_image = cover_image.astype(np.float32)
		cover_image = torch.from_numpy(cover_image)
		cover_image = torch.transpose(torch.transpose(cover_image, 2, 0), 1, 2)
		cover_image = cover_image / 255.0

		cover_th = cover_th.astype(np.float32).reshape(256,256,1)
		# print(cover_th.shape)
		cover_th = torch.from_numpy(cover_th)
		cover_th = torch.transpose(torch.transpose(cover_th, 2, 0), 1, 2)
		cover_th = cover_th / 255.0

		gt = gt.astype(np.float32)
		gt = torch.from_numpy(gt)
		gt = torch.transpose(torch.transpose(gt, 2, 0), 1, 2)
		gt = gt / 255.0

		return {'cover': cover_image,
				'msg': message,
				'th': cover_th,
				'gt':gt}


class Dataset_Load(Dataset):

	def __init__(self, msg_path, cover_path, transform=ToTensor()):
		self.msgdir = msg_path
		self.cover_dir = cover_path
		self.transform = transform

	def __len__(self):
		return cfg.NUM_IMAGES

	def __getitem__(self, index):

		message = cv2.imread(os.path.join(self.msgdir, os.listdir(self.msgdir)[index]))
		
		if message.shape[0] > cfg.IMAGE_SIZE[0] or message.shape[1] > cfg.IMAGE_SIZE[1]:
			message = cv2.resize(message,cfg.IMAGE_SIZE, interpolation = cv2.INTER_AREA)
		else:
			message = cv2.resize(message,cfg.IMAGE_SIZE, interpolation = cv2.INTER_CUBIC)
		message = message.reshape((cfg.IMAGE_SIZE[0], cfg.IMAGE_SIZE[1], cfg.NUM_CHANNELS))


		cover_im = cv2.imread(os.path.join(self.cover_dir, os.listdir(self.cover_dir)[index]))

		if cover_im.shape[0] > cfg.IMAGE_SIZE[0] or cover_im.shape[1] > cfg.IMAGE_SIZE[1]:
			cover_im = cv2.resize(cover_im,cfg.IMAGE_SIZE,interpolation = cv2.INTER_AREA)
		else:
			cover_im = cv2.resize(cover_im,cfg.IMAGE_SIZE,interpolation = cv2.INTER_CUBIC)
		cover_im = cover_im.reshape((cfg.IMAGE_SIZE[0], cfg.IMAGE_SIZE[1], cfg.NUM_CHANNELS))
		
		cover_th = threshold(cover_im)

		sample = { 'cover': cover_im, 'msg': message, 'th':cover_th, 'gt':cover_im}

		if self.transform != None:
			sample = self.transform(sample)

		return sample