import torch.nn as nn
from models.Utils import EqualizedLR_Conv2d, Pixel_norm, Minibatch_std

class FromRGB(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.conv = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=(1, 1))
		self.relu = nn.LeakyReLU(0.2)
		
	def forward(self, x):
		x = self.conv(x)
		return self.relu(x)

class ToRGB(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.conv = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(1,1), stride=(1, 1))
	
	def forward(self, x):

		return self.conv(x)
        
class G_Block(nn.Module):
	def __init__(self, in_ch, out_ch, initial_block=False):
		super().__init__()
		if initial_block:
			self.upsample = None
			self.conv1 = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(4, 4), stride=(1, 1), padding=(3, 3))
		else:
			self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
			self.conv1 = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.conv2 = EqualizedLR_Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu = nn.LeakyReLU(0.2)
		self.pixelwisenorm = Pixel_norm()
		nn.init.normal_(self.conv1.weight)
		nn.init.normal_(self.conv2.weight)
		nn.init.zeros_(self.conv1.bias)
		nn.init.zeros_(self.conv2.bias)

	def forward(self, x):

		if self.upsample is not None:
			x = self.upsample(x)
		# x = self.conv1(x*scale1)
		x = self.conv1(x)
		x = self.relu(x)
		x = self.pixelwisenorm(x)
		# x = self.conv2(x*scale2)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.pixelwisenorm(x)
		return x

class D_Block(nn.Module):
	def __init__(self, in_ch, out_ch, initial_block=False):
		super().__init__()

		if initial_block:
			self.minibatchstd = Minibatch_std()
			self.conv1 = EqualizedLR_Conv2d(in_ch+1, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			self.conv2 = EqualizedLR_Conv2d(out_ch, out_ch, kernel_size=(4, 4), stride=(1, 1))
			self.outlayer = nn.Sequential(
									nn.Flatten(),
									nn.Linear(out_ch, 1)
									)
		else:			
			self.minibatchstd = None
			self.conv1 = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			self.conv2 = EqualizedLR_Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			self.outlayer = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

		self.relu = nn.LeakyReLU(0.2)
		nn.init.normal_(self.conv1.weight)
		nn.init.normal_(self.conv2.weight)
		nn.init.zeros_(self.conv1.bias)
		nn.init.zeros_(self.conv2.bias)
	
	def forward(self, x):
		if self.minibatchstd is not None:
			x = self.minibatchstd(x)
		
		x = self.conv1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.outlayer(x)
		return x
