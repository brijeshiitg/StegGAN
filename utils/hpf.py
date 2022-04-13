import torch
from torch.autograd import Variable
from torch.nn.functional import conv2d

from config import cfg


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hpf(x):
	KV = torch.tensor([[-1,2,-2,2,-1],
		   [2,-6,8,-6,2],
		   [-2,8,-12,8,-2],
		   [2,-6,8,-6,2],
		   [-1,2,-2,2,-1]])/12.
	KV = KV.view(1,1,5,5).to(device=device, dtype=torch.float)
	KV = torch.autograd.Variable(KV, requires_grad=False)

	# with torch.no_grad():
	out = conv2d(x.view(-1, 1, x.shape[2], x.shape[3]), KV, padding=2).view(cfg.BATCH_SIZE, cfg.NUM_CHANNELS, x.shape[2], x.shape[3]).to(device=device)
	return out