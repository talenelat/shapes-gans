from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.utils as vutils


class Generator(nn.Module):
    def __init__(self, nz, nc, ngf, dropout_rate, image_size, normalization):
        super(Generator, self).__init__()

        self.nz = nz
        self.nc = nc
        self.ngf = ngf
        self.dropout_rate = dropout_rate
        self.image_size = image_size
        self.normalization = normalization # Error 1: didn't initialize
        
        # Input: (nz)
        
        self.gen_ct1 = nn.ConvTranspose2d(in_channels=self.nz, out_channels=2 * self.ngf, 
                                          kernel_size=self.image_size//8, stride=1, padding=0, bias=False)
        self.gen_bn1 = nn.BatchNorm2d(2 * self.ngf)
        self.gen_relu1 = nn.ReLU(True)
        self.gen_drop1 = nn.Dropout2d(self.dropout_rate)
        # Output: (2 * ngf x image_size // 8 x image_size // 8)
        
        self.gen_ct2 = nn.ConvTranspose2d(in_channels=2 * self.ngf, out_channels=self.ngf, 
                                          kernel_size=6, stride=4, padding=1, bias=False)
        self.gen_bn2 = nn.BatchNorm2d(self.ngf)
        self.gen_relu2 = nn.ReLU(True)
        self.gen_drop2 = nn.Dropout2d(self.dropout_rate)
        # Output (ngf x image_size // 2 x image_size // 2)
        
        self.gen_ct4 = nn.ConvTranspose2d(in_channels=self.ngf, out_channels=nc, 
                                          kernel_size=4, stride=2, padding=1, bias=False)
        self.gen_tanh = nn.Tanh()
        # Output (nc x image_size x image_size)
    

    def forward(self, z):
        if self.normalization == True:
            g1 = self.gen_drop1(self.gen_relu1(self.gen_bn1(self.gen_ct1(z))))
            g2 = self.gen_drop2(self.gen_relu2(self.gen_bn2(self.gen_ct2(g1))))
        else:
            g1 = self.gen_drop1(self.gen_relu1(self.gen_ct1(z)))
            g2 = self.gen_drop2(self.gen_relu2(self.gen_ct2(g1)))
        g3 = self.gen_tanh(self.gen_ct4(g2))
        return g3


class Discriminator(nn.Module):
    '''
    DCGAN model Discriminator.
    '''
    def __init__(self, nz, nc, ndf, image_size, normalization):
        super(Discriminator, self).__init__()

        self.nz = nz
        self.nc = nc
        self.ndf = ndf
        self.image_size = image_size
        self.normalization = normalization

        # Input: (nc x image_size x image_size)
        
        self.disc_cv1 = nn.Conv2d(in_channels=self.nc, out_channels=self.ndf, 
                                  kernel_size=3, stride=1, padding=1, bias=False)
        self.disc_lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        # Output: (ndf x image_size x image_size)
        
        self.disc_cv2 = nn.Conv2d(in_channels=self.ndf, out_channels=self.ndf * 2, 
                                  kernel_size=4, stride=2, padding=1, bias=False)
        self.disc_bn1 = nn.BatchNorm2d(self.ndf * 2)
        self.disc_lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        # Output: (2 * ndf x image_size // 2 x image_size // 2)

        self.disc_cv3 = nn.Conv2d(in_channels=self.ndf * 2, out_channels=self.ndf * 4, 
                                  kernel_size=6, stride=4, padding=1, bias=False)
        self.disc_bn2 = nn.BatchNorm2d(self.ndf * 4)
        self.disc_lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        # Output: (4 * ndf x image_size // 8 x image_size // 8)
        
        self.disc_cv4 = nn.Conv2d(in_channels=self.ndf * 4, out_channels= 1, 
                                  kernel_size=self.image_size // 8, stride=1, padding=0, bias=False)
        self.disc_sig = nn.Sigmoid()
        # Output: (1 x 1 x 1)
    
    def forward(self, image):
        d1 = self.disc_lrelu1(self.disc_cv1(image))
        if self.normalization == True:
            d2 = self.disc_lrelu2(self.disc_bn1(self.disc_cv2(d1)))
            d3 = self.disc_lrelu3(self.disc_bn2(self.disc_cv3(d2)))
        else:
            d2 = self.disc_lrelu2(self.disc_cv2(d1))
            d3 = self.disc_lrelu3(self.disc_cv3(d2))
        d4 = self.disc_sig(self.disc_cv4(d3))
        return d4.view(-1, 1).squeeze(1)