import torch
from torch.autograd import Variable

def normalize_imagenet(batch):
    tensortype = type(batch.data)
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 0.40760392
    mean[:, 1, :, :] = 0.45795686
    mean[:, 2, :, :] = 0.48501960

    stdv = tensortype(batch.data.size())
    stdv[:, 0, :, :] = 0.225
    stdv[:, 1, :, :] = 0.224
    stdv[:, 2, :, :] = 0.229

    return torch.div(torch.sub(batch , Variable(mean).cuda()), Variable(stdv).cuda())