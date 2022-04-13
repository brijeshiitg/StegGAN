import torch
import torch.nn as nn

from config import cfg


class Discriminator_ex(nn.Module):

	def __init__(self):
		super(Discriminator_ex, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(num_features=8)
		self.prelu1 = nn.PReLU(num_parameters=8)

		self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(num_features=16)
		self.prelu2 = nn.PReLU(num_parameters=16)

		self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(num_features=16)
		self.prelu3 = nn.PReLU(num_parameters=16)

		self.conv4 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
		self.bn4 = nn.BatchNorm2d(num_features=1)
		self.prelu4 = nn.PReLU(num_parameters=1)

	def forward(self, input):

		layer1 = self.prelu1(self.bn1(self.conv1(input)))
		layer2 = self.prelu2(self.bn2(self.conv2(layer1)))
		layer3 = self.prelu3(self.bn3(self.conv3(layer2)))
		layer4 = self.prelu4(self.bn4(self.conv4(layer3)))

		final_logits = torch.mean(torch.sigmoid(layer4.view(cfg.BATCH_SIZE, -1)), 1)
		
		return final_logits