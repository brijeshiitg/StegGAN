import os
import re

import cv2

import numpy as np


def latest_checkpoint(checkpoints_dir):
	''' If you get ValueError: max() arg is an empty sequence. Means your checkpoints_dir is empty just delete it.'''
	if os.path.exists(checkpoints_dir):
		if len(os.listdir(checkpoints_dir)) > 0:
			all_chkpts = "".join(os.listdir(checkpoints_dir))
			latest = max(map(int, re.findall('\d+', all_chkpts)))
		else:
			latest = None
	else:
		latest = None
	return latest


def threshold(img):
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	_,im1 = cv2.threshold(np.array(img),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	# im1 = cv2.cvtColor(im1,cv2.COLOR_GRAY2RGB)
	return im1


def time_taken(tt):
	if tt>60:
		tt=round(tt/60.,2)
		return str(tt)+'mins.'
	elif tt>3600:
		tt=round(tt/3600.,2)
		return str(tt)+'hrs.'
	else: return str(round(tt,2))+'sec.'
