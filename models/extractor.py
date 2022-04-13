import torch
import torch.nn as nn


class Extractor(nn.Module):

	def __init__(self):
		super(Extractor, self).__init__()

		# layers for encoder network

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(num_features=32)
		self.prelu1 = nn.PReLU(num_parameters=32)

		self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(num_features=32)
		self.prelu2 = nn.PReLU(num_parameters=32)

		self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(num_features=32)
		self.prelu3 = nn.PReLU(num_parameters=32)

		self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.bn4 = nn.BatchNorm2d(num_features=32)
		self.prelu4 = nn.PReLU(num_parameters=32)

		self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.bn5 = nn.BatchNorm2d(num_features=32)
		self.prelu5 = nn.PReLU(num_parameters=32)

		self.conv6 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
		self.bn6 = nn.BatchNorm2d(num_features=3)
		self.prelu6 = nn.PReLU(num_parameters=3)

		# layers for decoder network
		self.d_conv1 = nn.ConvTranspose2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.d_bn1 = nn.BatchNorm2d(num_features=32)
		self.d_relu1 = nn.ReLU()

		self.d_conv2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.d_bn2 = nn.BatchNorm2d(num_features=32)
		self.d_relu2 = nn.ReLU()

		self.d_conv3 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.d_bn3 = nn.BatchNorm2d(num_features=32)
		self.d_relu3 = nn.ReLU()

		self.d_conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.d_bn4 = nn.BatchNorm2d(num_features=32)
		self.d_relu4 = nn.ReLU()

		self.d_conv5 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.d_bn5 = nn.BatchNorm2d(num_features=32)
		self.d_relu5 = nn.ReLU()

		self.d_conv6 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
		self.d_bn6 = nn.BatchNorm2d(num_features=3)
		self.d_relu6 = nn.ReLU()

	def forward(self, input):

		# encoder

		# input = torch.cat((input_cover, input_message), 1)
		en_layer1 = self.prelu1(self.bn1(self.conv1(input)))
		en_layer2 = self.prelu2(self.bn2(self.conv2(en_layer1)))
		en_layer3 = self.prelu3(self.bn3(self.conv3(en_layer2)))
		en_layer4 = self.prelu4(self.bn4(self.conv4(en_layer3)))
		en_layer5 = self.prelu5(self.bn5(self.conv5(en_layer4)))
		en_layer6 = self.prelu6(self.bn6(self.conv6(en_layer5)))

		# decoder
		de_layer1 = self.d_relu1(self.d_bn1(self.d_conv1(en_layer6)))
		de_layer2 = self.d_relu2(self.d_bn2(self.d_conv2(de_layer1)))

		# skip connection
		de_layer2 = de_layer2 + en_layer2
		de_layer3 = self.d_relu3(self.d_bn3(self.d_conv3(de_layer2)))
		de_layer4 = self.d_relu4(self.d_bn4(self.d_conv4(de_layer3)))

		# skip connection
		de_layer4 = de_layer4 + en_layer4
		de_layer5 = self.d_relu5(self.d_bn5(self.d_conv5(de_layer4)))
		de_layer6 = self.d_relu6(self.d_bn6(self.d_conv6(de_layer5)))

		return de_layer6