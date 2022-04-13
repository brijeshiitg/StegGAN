''' training gan with xunet as discriminator for generator and an extractor with discriminator. num_images=17496 to 
make it divisible by batch_size.'''
import argparse
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import logging
import time

from utils.utils import threshold, time_taken, latest_checkpoint
from utils.hpf import hpf
from utils.normalize import normalize_imagenet
from utils.vgg16 import VGG16_intermediate


from models.embedder import Embedder 
from models.extractor import Extractor 
from models.discriminator_em import Steganalyzer
from models.discriminator_ex import Discriminator_ex

from config import cfg
from dataset import Dataset_Load

cfg.merge_from_file('./experiment.yaml')

logging.basicConfig(filename='training.log',format='%(asctime)s %(message)s', level=logging.DEBUG)


print("Embedder learning rate: ", cfg.EM_LR)
print("Steganalyzer learning rate: ", cfg.ST_LR)
print("Extractor  learning rate: ", cfg.EX_LR)
print("Extractor's discriminator learning rate: ", cfg.DI_LR)
print("\n")



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

	batches = int(cfg.NUM_IMAGES/cfg.BATCH_SIZE)

	real_label = 1
	fake_label = 0

	# Embedder network
	netG = Embedder()
	mse_loss = nn.MSELoss()
	netG.to(device)
	optim_g = optim.Adam(netG.parameters(), lr=cfg.EM_LR)

	# Embedder's discriminator netowrk
	netD = Steganalyzer()
	bce_loss = nn.BCELoss()
	netD.to(device)
	optim_d = torch.optim.SGD(netD.parameters(), cfg.ST_LR, momentum = 0.9)

	# VGG features for perceptual loss
	vgg = VGG16_intermediate()
	vgg.to(device)
	vgg.eval()
	for p in vgg.parameters():
		p.requires_grad = False

	# Extractor network
	netE = Extractor()
	e_mse_loss = nn.MSELoss()
	netE.to(device)
	optim_e = optim.Adam(netE.parameters(), lr=cfg.EX_LR)

	# Extractor's discriminator network
	netDX = Discriminator_ex()
	bcex_loss = nn.BCELoss()
	netDX.to(device)
	optim_dx = optim.Adam(netDX.parameters(), lr=cfg.DI_LR)

	# dataset and dataloader
	dataset = Dataset_Load(msg_path=cfg.SECRET_PATH, cover_path=cfg.COVER_PATH)
	dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)


	latest = latest_checkpoint(cfg.CHECKPOINT_PATH)

	if latest is None:
		start_epoch = 1

		print("No checkpoints found!! \nRetraining...\n")
		logging.debug("No checkpoints found!!\nRetraining\n")
		if not os.path.exists(cfg.CHECKPOINT_PATH):
			os.makedirs(cfg.CHECKPOINT_PATH)	
	else:
		checkpoint_g = torch.load(os.path.join(cfg.CHECKPOINT_PATH, "netG_"+str(latest)+".pt"))
		checkpoint_d = torch.load(os.path.join(cfg.CHECKPOINT_PATH, "netD_"+str(latest)+".pt"))
		checkpoint_e = torch.load(os.path.join(cfg.CHECKPOINT_PATH, "netE_"+str(latest)+".pt"))
		checkpoint_dx = torch.load(os.path.join(cfg.CHECKPOINT_PATH, "netDX_"+str(latest)+".pt"))
		
		start_epoch = checkpoint_g['epoch'] + 1
		
		netG.load_state_dict(checkpoint_g['model_state_dict'])
		optim_g.load_state_dict(checkpoint_g['optimizer_state_dict'])
		
		netD.load_state_dict(checkpoint_d['model_state_dict'])
		optim_d.load_state_dict(checkpoint_d['optimizer_state_dict'])

		netE.load_state_dict(checkpoint_e['model_state_dict'])
		optim_e.load_state_dict(checkpoint_e['optimizer_state_dict'])

		netDX.load_state_dict(checkpoint_dx['model_state_dict'])
		optim_dx.load_state_dict(checkpoint_dx['optimizer_state_dict'])
		
		print("Restoring model from checkpoint " + str(start_epoch))
		logging.debug("Restoring model from checkpoint " + str(start_epoch))

	netG.train()
	netD.train()
	netE.train()
	netDX.train()

	for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):

		total_Emse_loss = 0.0
		total_G_loss = 0.0
		total_mse_loss = 0.0
		total_EP_loss = 0.0
		total_D_loss = 0.0
		total_E_loss = 0.0
		total_DX_loss = 0.0
		total_adv_loss = 0.0


		s1=time.time() 
		for i_batch, sample_batched in enumerate(dataloader):
			s= time.time()
			
			cover_batch = sample_batched['cover'].to(device)
			msg_batch = sample_batched['msg'].to(device)
			th_batch = sample_batched['th'].to(device)
			gt_batch = sample_batched['gt'].to(device)

			# max ED first
			for p in netDX.parameters():
				p.requires_grad = True

			optim_dx.zero_grad()

			for p in netE.parameters():
				p.requires_grad = False

			for p in netG.parameters():
				p.requires_grad = False    

			dx_real_logit = netDX(msg_batch)

			dx_real_label = torch.full((cfg.BATCH_SIZE,), real_label, device=device)

			dx_loss_real = bce_loss(dx_real_logit, dx_real_label)
			dx_loss_real.backward(retain_graph=True) 

			stego_batch = netG(cover_batch, msg_batch, th_batch)
			pred_msg = netE(stego_batch)

			dx_fake_logit = netDX(pred_msg)
			dx_fake_label = torch.full((cfg.BATCH_SIZE,), fake_label, device=device)

			dx_loss_fake = bce_loss(dx_fake_logit, dx_fake_label)
			dx_loss_fake.backward()
			
			optim_dx.step()          
			
			batch_DX_loss = float(dx_loss_fake) + float(dx_loss_real)  # Discriminator of Extractor loss
			total_DX_loss += batch_DX_loss
			# max ED done

			for p in netE.parameters():
				p.requires_grad = True

			optim_e.zero_grad()

			ext_msg_pred = netE(stego_batch)
			batch_Emse_loss = e_mse_loss(ext_msg_pred, msg_batch)
			batch_Emse_loss.backward(retain_graph=True)
			total_Emse_loss += batch_Emse_loss.item()
			# print("======",netE.conv1.weight.grad[0][0])


			for p in netDX.parameters():
				p.requires_grad = False

			dx_real_logit = netDX(ext_msg_pred)
			dx_real_label = torch.full((cfg.BATCH_SIZE,), real_label, device=device)            

			dx_loss_real = torch.mul(cfg.WT_EXT_ADV, bce_loss(dx_real_logit, dx_real_label))
			dx_loss_real.backward(retain_graph=True) 
			# print("******",netE.conv1.weight.grad[0][0])

			feats_orig_msg = vgg( normalize_imagenet(msg_batch))
			feats_pred_msg = vgg( normalize_imagenet(ext_msg_pred))

			ext_perceptual = torch.mul(cfg.WT_PERCEPT, mse_loss(feats_pred_msg[1], feats_orig_msg[1]))
			ext_perceptual.backward()

			batch_EP_loss = ext_perceptual.item()
			total_EP_loss += batch_EP_loss
			# print("-------",netE.conv1.weight.grad[0][0])

			optim_e.step()

			batch_E_loss = float(batch_Emse_loss) + float(dx_loss_real) + batch_EP_loss #Extractor loss
			total_E_loss += batch_E_loss
			# Extractor done

			for p in netD.parameters():
				p.requires_grad = True

			optim_d.zero_grad()

			for p in netG.parameters():
				p.requires_grad = False

			c_batch = hpf(cover_batch)    
			d_real_logit = netD(c_batch)
			d_real_label = torch.full((cfg.BATCH_SIZE,), real_label, device=device)

			d_real_logit = torch.max(d_real_logit, 1)[0].float()


			d_loss_real = bce_loss(torch.sub(1,d_real_logit), d_real_label)
			d_loss_real.backward(retain_graph=True) 

			fake_images = netG(cover_batch, msg_batch, th_batch)
			fake_images = hpf(fake_images)    
			d_fake_logit = netD(fake_images)
			d_fake_logit =  torch.max(d_fake_logit, 1)[0].float()
			d_fake_label = torch.full((cfg.BATCH_SIZE,), fake_label, device=device)

			d_loss_fake = bce_loss(torch.sub(1,d_fake_logit), d_fake_label)
			d_loss_fake.backward()
			
			optim_d.step()          
			
			batch_D_loss = float(d_loss_fake) + float(d_loss_real)  #Discriminator loss
			total_D_loss += batch_D_loss


			# min G now

			for p in netG.parameters():
				p.requires_grad = True

			optim_g.zero_grad()
			
			# get stego images
			pred_batch = netG(cover_batch, msg_batch, th_batch)
			
			# mse loss
			batch_mse_loss = torch.mul(cfg.WT_MSE, mse_loss(pred_batch, cover_batch))
			batch_mse_loss.backward(retain_graph=True)
			
			batch_mse_loss = float(batch_mse_loss)
			total_mse_loss+=batch_mse_loss

			# for adv loss
			for p in netD.parameters():
				p.requires_grad = False

			pred_batch_f = hpf(pred_batch)    

			stego_logits = netD(pred_batch_f)
			stego_logits =  torch.max(stego_logits, 1)[0].float()
			stego_labels = torch.full((cfg.BATCH_SIZE,), real_label, device=device)

			batch_adv_loss = torch.mul(cfg.WT_ADV, bce_loss(torch.sub(1,stego_logits), stego_labels))
			batch_adv_loss.backward(retain_graph=True)

			batch_adv_loss = float(batch_adv_loss)
			total_adv_loss += batch_adv_loss
		
			# min E loss
			for p in netE.parameters():
				p.requires_grad = False

			stego = netG(cover_batch, msg_batch, th_batch)
			ext_msg_pred = netE(stego)
			batch_E_loss_g = torch.mul(cfg.WT_EXT, e_mse_loss(ext_msg_pred, msg_batch)) #extractor loss going to generator
			batch_E_loss_g.backward()
			
			# optim G
			optim_g.step()

			batch_G_loss = batch_mse_loss + batch_adv_loss + float(batch_E_loss_g)
			total_G_loss += batch_G_loss

			nb = float(cfg.NUM_IMAGES / cfg.BATCH_SIZE) #total no. of batches
			f=time.time()
			batch_tt = f-s

			print( '| Ep: %d| %d/%d batch| G_mse: %.6f| G_adv: %.6f| G_loss: %.6f| D_loss: %.6f| E_mse: %.6f| E_adv: %.6f| E_loss: %.6f| EP_loss:%.6f| DX_loss: %.6f|'
				% (epoch, i_batch, batches, batch_mse_loss, batch_adv_loss, batch_G_loss, batch_D_loss, batch_Emse_loss, dx_loss_real, batch_E_loss, batch_EP_loss, batch_DX_loss))


		f1 = time.time()
		epoch_tt = f1-s1
		print( '\n| Ep: %d over | Gen_loss: %.6f | G_MSE: %.6f |percept_loss: %.6f | Disc_loss: %.6f | E_loss: %.6f | E_mse:%.6f | DisEx_loss: %.6f | T_ep:%s |\n'
			% (epoch, total_G_loss/nb, total_mse_loss/nb, total_EP_loss/nb, total_D_loss/nb, total_E_loss/nb,total_Emse_loss/nb, total_DX_loss/nb, time_taken(epoch_tt)))
		logging.debug( '\n| Epoch: %d over | Gen_loss: %.6f | G_MSE: %.6f | perc_loss: %.6f | Disc_loss: %.6f | E_loss: %.6f |E_mse: %.6f | DisEx_loss: %.6f |T_ep:%s |\n'
			% (epoch, total_G_loss/nb, total_mse_loss/nb, total_EP_loss/nb, total_D_loss/nb, total_E_loss/nb, total_Emse_loss/nb, total_DX_loss/nb, time_taken(epoch_tt)))

		torch.save({
			'epoch' : epoch,
			'model_state_dict' : netG.state_dict(),
			'optimizer_state_dict' : optim_g.state_dict(),
			'mse_loss': total_mse_loss,
			'entropy_loss': total_adv_loss,
			'total_loss': total_G_loss
		}, os.path.join(cfg.CHECKPOINT_PATH, 'netG_'+str(epoch) + '.pt'))

		torch.save({
			'epoch' : epoch,
			'model_state_dict' : netD.state_dict(),
			'optimizer_state_dict' : optim_d.state_dict(),
			'entropy_loss': total_D_loss
		}, os.path.join(cfg.CHECKPOINT_PATH, 'netD_'+ str(epoch) + '.pt'))

		torch.save({
			'epoch' : epoch,
			'model_state_dict' : netE.state_dict(),
			'optimizer_state_dict' : optim_e.state_dict(),
			'e_mse_loss': total_E_loss,
			'e_perceptual_loss': total_EP_loss
		}, os.path.join(cfg.CHECKPOINT_PATH, 'netE_'+ str(epoch) + '.pt'))

		torch.save({
			'epoch' : epoch,
			'model_state_dict' : netDX.state_dict(),
			'optimizer_state_dict' : optim_dx.state_dict(),
			'dx_entropy_loss': total_DX_loss
		}, os.path.join(cfg.CHECKPOINT_PATH, 'netDX_'+ str(epoch) + '.pt'))
