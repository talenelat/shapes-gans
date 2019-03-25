import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision


class Encoder(nn.Module):
    def __init__(self, nc, nz, ndf, image_size, cuda=False, normalization=False):
        super(Encoder, self).__init__()

        self.nc = nc
        self.nz = nz
        self.ndf = ndf
        self.image_size = image_size
        self.cuda = cuda
        self.normalization = normalization

        # Input: (nc x image_size x image_size)

        self.enc_cv1 = nn.Conv2d(self.nc, self.ndf, 4, 2, 1)
        self.enc_le1 = nn.LeakyReLU(0.2, inplace=True)
        self.enc_bn1 = nn.BatchNorm2d(self.ndf)
        # Output: (ndf x image_size // 2 x image_size // 2)

        self.enc_cv2 = nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1)
        self.enc_le2 = nn.LeakyReLU(0.2, inplace=True)
        self.enc_bn2 = nn.BatchNorm2d(self.ndf*2)
        # Output: (2*ndf x image_size // 4 x image_size // 4)

        self.enc_cv3 = nn.Conv2d(self.ndf*2, self.ndf*4, 4, 2, 1)
        self.enc_le3 = nn.LeakyReLU(0.2, inplace=True)
        self.enc_bn3 = nn.BatchNorm2d(self.ndf*4)
        # Output: (4*ndf x image_size // 8 x image_size // 8)

        self.enc_fc_mu = nn.Linear(self.ndf*4 * (self.image_size//8) * (self.image_size//8), nz)
        self.enc_fc_log = nn.Linear(self.ndf*4 * (self.image_size//8) * (self.image_size//8), nz)
        # Output: (nz)

    def calculate_z(self, mu, logvar):
        std = torch.exp(logvar / 2)
        if self.cuda == True:
            eps = torch.FloatTensor(std.size()).normal_().cuda()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        z = eps * std + mu
        return z

    def forward(self, image):
        if self.normalization == True:
            e1 = self.enc_bn1(self.enc_le1(self.enc_cv1(image)))
            e2 = self.enc_bn2(self.enc_le2(self.enc_cv2(e1)))
            e3 = self.enc_bn3(self.enc_le3(self.enc_cv3(e2)))
        else:
            e1 = self.enc_le1(self.enc_cv1(image))
            e2 = self.enc_le2(self.enc_cv2(e1))
            e3 = self.enc_le3(self.enc_cv3(e2))
        e3 = e3.view(-1, self.ndf*4*4*4)

        mu = self.enc_fc_mu(e3)
        logvar = self.enc_fc_log(e3)

        z = self.calculate_z(mu=mu, logvar=logvar)
        return z


class Decoder(nn.Module):
    def __init__(self, nz, nc, ngf, image_size, normalization=False):
        super(Decoder, self).__init__()

        self.nc = nc
        self.nz = nz
        self.ngf = ngf
        self.image_size = image_size
        self.normalization = normalization

        # Input: (nz)

        self.dec_ln1 = nn.Linear(self.nz, 4*self.ngf * self.image_size // 8 * self.image_size // 8)
        self.dec_lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        # -> Transform
        # Output: (4*ngf * image_size // 8 * image_size // 8)
        
        self.dec_ct1 = nn.ConvTranspose2d(4*self.ngf, 2*self.ngf, 4, 2, 1)
        self.dec_lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.dec_bn1 = nn.BatchNorm2d(2*self.ngf)
        # Output: (2*ngf * image_size // 4 * image_size // 4)
       
        self.dec_ct2 = nn.ConvTranspose2d(2*self.ngf, self.ngf, 4, 2, 1)
        self.dec_lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        self.dec_bn2 = nn.BatchNorm2d(self.ngf, 1.e-3)
        # Output: (ngf * image_size // 2 * image_size // 2)
        
        self.dec_ct3 = nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1)
        self.sigmoid = nn.Sigmoid()
        # Output: (ngf x image_size x image_size)

    def forward(self, z):
        d1 = self.dec_lrelu1(self.dec_ln1(z))
        d1 = d1.view(-1, self.ngf*4, 4, 4)
        if self.normalization == True:
            d2 = self.dec_bn1(self.dec_lrelu2(self.dec_ct1(d1)))
            d3 = self.dec_bn2(self.dec_lrelu3(self.dec_ct2(d2)))
        else:
            d2 = self.dec_lrelu2(self.dec_ct1(d1))
            d3 = self.dec_lrelu3(self.dec_ct2(d2))
        d4 = self.dec_ct3(d3)
        d5 = self.sigmoid(d4)
        return d5