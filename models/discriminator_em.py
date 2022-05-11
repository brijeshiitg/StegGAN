import torch
import torch.nn as nn

from config import cfg


class Steganalyzer(nn.Module):
	"""Steganalyzer for Generator"""
	def __init__(self):
		super(Steganalyzer, self).__init__()
		# convolutional layer
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2)
		self.bn1 = nn.BatchNorm2d(num_features=8) 
		self.avgpool = nn.AvgPool2d(5,stride=2, padding=2)

		self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2)
		self.bn2 = nn.BatchNorm2d(num_features=16)
		self.tanh2 = nn.Tanh()

		self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1)
		self.bn3 = nn.BatchNorm2d(num_features=32)
		self.relu3 =nn.ReLU()

		self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1)
		self.bn4 = nn.BatchNorm2d(num_features=64)
		self.relu4 =nn.ReLU()

		self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1)
		self.bn5 = nn.BatchNorm2d(num_features=128)
		self.relu5 =nn.ReLU()
		self.avgpool1 = nn.AvgPool2d(16,stride=2)

		self.fc = nn.Linear(128*1*1, 2)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x): # x is input
		# add sequence of convolutional and max-pooling layers
		x = self.avgpool(torch.tanh(self.bn1(torch.abs(self.conv1(x)))))
		# print('group1: ', x.size())
		x = self.avgpool(torch.tanh(self.bn2(self.conv2(x))))
		# print('group2: ', x.size())
		x = self.avgpool(self.relu3(self.bn3(self.conv3(x))))
		# print('group3: ', x.size())
		x = self.avgpool(self.relu4(self.bn4(self.conv4(x))))
		# print('group4: ', x)
		x = self.avgpool1(self.relu5(self.bn5(self.conv5(x))))
		# print('group5: ', x.size())
		x = x.view(cfg.BATCH_SIZE,-1)
		x = self.softmax(self.fc(x))

		# index = torch.max(x, 1)
		# print('fc: ',x)
		return x