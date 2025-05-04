
import torch
import numpy as np
from torch.nn import Module

image_size = 256
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def low_pass_filter(image, r = 8):
    f = torch.tensor(image)
    f = torch.fft.fft2(f)
    f_compress = torch.complex(torch.zeros(in_channels,2*r,r), torch.zeros(in_channels,2*r,r))
    for i in range(in_channels):
        f_compress[i, :r, :r] = f[i, :r, :r]
        f_compress[i, -r:, :r] = f[i, -r:, :r]
    return f_compress


def low_pass_filter2(image, r = 8, in_channels = 1, image_size = 256):
    f = image
    f_compress = torch.complex(torch.zeros(in_channels, image_size, image_size//2), torch.zeros(in_channels, image_size, image_size//2))
    for i in range(in_channels):
        f_compress[i, :r, :r] = f[i, :r, :r]
        f_compress[i, -r:, :r] = f[i, -r:, :r]
    return f_compress


def low_pass_filter3(image, r = 8, in_channels = 1, image_size = 256):
    f = image
    f_compress = torch.complex(torch.zeros(f.size(0),1,image_size, image_size//2,2), torch.zeros(f.size(0),1,image_size, image_size//2,2))
    f_compress[:,:, :r, :r,:] = f[:,:, :r, :r,:]
    f_compress[:,:, -r:, :r,:] = f[:,:, -r:, :r,:]
    return f_compress

def low_pass_filter4(image, r = 8, in_channels = 1, image_size = 256):
    f = image
    f_compress = torch.complex(torch.zeros(f.size(0),1,image_size, image_size//2), torch.zeros(f.size(0),1,image_size, image_size//2))
    f_compress[:,:, :r, :r] = f[:,:, :r, :r]
    f_compress[:,:, -r:, :r] = f[:,:, -r:, :r]
    return f_compress


def normalize_image(img):
    img_min = np.min(img)
    img_max = np.max(img)
    normalized_img = (img - img_min) / (img_max - img_min)
    return normalized_img

class LowPassFilter(Module):
    def __init__(self, r, in_channels = 1):
        super(LowPassFilter,self).__init__()
        self.r = r
        self.fft = torch.fft.fft2
        self.in_channels = in_channels
    def forward(self, x):
        r = self.r

        f = self.fft(x)
        f_compress = torch.complex(torch.zeros(x.shape[0], self.in_channels, 2*r, r), torch.zeros(x.shape[0], self.in_channels, 2*r, r))
        assert self.in_channels==x.shape[1], f"data input_channels : {x.shape[1]} not equals to models input_channels : {self.in_channels}"
            
        for i in range(self.in_channels):
            f_compress[:,i, :r, :r] = f[:, i, :r, :r]
            f_compress[:, i, -r:, :r] = f[:, i, -r:, :r]
        return f_compress

class ZeroPadding(Module):
    def __init__(self, size, r):
        super(ZeroPadding,self).__init__()
        self.r = r
        self.size = size
    def forward(self, f):
        r = self.r

        f_compress = torch.complex(torch.zeros(f.size(0),1,self.size,self.size), torch.zeros(f.size(0),1,self.size, self.size))
        f_compress[:, :, :r, :r] = f[:,:, :r, :r]
        f_compress[:,:, -r:, :r] = f[:,:, -r:, :r]
        return f_compress