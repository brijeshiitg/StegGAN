import torch.nn as nn
from torchvision import models


# Taking intermediate features of VGG16
orig_vgg16 = models.vgg16(pretrained=True)

class VGG16_intermediate(nn.Module):
    def __init__(self):
        super(VGG16_intermediate, self).__init__()
        self.feat1 = nn.Sequential(*list(orig_vgg16.features.children())[:4])
        self.feat2 = nn.Sequential(*list(orig_vgg16.features.children())[:9])

    def forward(self, x):
        f1 = self.feat1(x)
        f2 = self.feat2(x)
        return [f1, f2]

# if __name__ == "__main__":
    # model = VGG16_intermediate()
    # print(model)