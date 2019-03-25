from __future__ import print_function
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, nz, nc, image_size, normalization):
        super(Generator, self).__init__()

        self.nz = nz
        self.nc = nc
        self.image_size = image_size
        self.normalization = normalization

        # Input: (nz)
        self.gen_lin1 = nn.Linear(self.nz, 
                                 self.nc * self.image_size * self.image_size // 4)
        self.gen_lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.gen_bn1 = nn.BatchNorm1d(self.nc * self.image_size * self.image_size // 4)
        # Output: (nc * image_size * image_size // 4)

        self.gen_lin2 = nn.Linear(self.nc * self.image_size * self.image_size // 4, 
                                  self.nc * self.image_size * self.image_size // 2)
        self.gen_lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.gen_bn2 = nn.BatchNorm1d(self.nc * self.image_size * self.image_size // 2)
        # Output: (nc * image_size * image_size // 2)

        self.gen_lin3 = nn.Linear(self.nc * self.image_size * self.image_size // 2,
                                  self.nc * self.image_size * self.image_size)
        self.gen_tanh = nn.Tanh()
        # Output: (nc * image_size * image_size)
    
    def forward(self, z):
        z = z.view(-1, self.nz)
        if self.normalization == True:
            g1 = self.gen_bn1(self.gen_lrelu1(self.gen_lin1(z)))
            g2 = self.gen_bn2(self.gen_lrelu2(self.gen_lin2(g1)))
        else:             
            g1 = self.gen_lrelu1(self.gen_lin1(z))
            g2 = self.gen_lrelu2(self.gen_lin2(g1))
        g3 = self.gen_tanh(self.gen_lin3(g2))

        return g3.view(-1, self.nc, self.image_size, self.image_size)


class Discriminator(nn.Module):
    def __init__(self, nz, nc, image_size):
        super(Discriminator, self).__init__()

        self.nz = nz
        self.nc = nc
        self.image_size = image_size

        # Input: (self.nc * self.image_size * self.image_size)

        self.disc_fc1 = nn.Linear(self.nc * self.image_size * self.image_size,
                                  self.nc * self.image_size * self.image_size // 2)
        self.disc_lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        # Output: (self.nc * self.image_size * self.image_size // 2)

        self.disc_fc2 = nn.Linear(self.nc * self.image_size * self.image_size // 2,
                                  self.nc * self.image_size * self.image_size // 4)
        self.disc_lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        # Output: (self.nc * self.image_size * self.image_size // 4)

        self.disc_fc3 = nn.Linear(self.nc * self.image_size * self.image_size // 4, 1)
        self.disc_sig = nn.Sigmoid()
        # Output: (nz)

    def forward(self, image):
        image = image.view(-1, self.nc*self.image_size*self.image_size)
        d1 = self.disc_lrelu1(self.disc_fc1(image))
        d2 = self.disc_lrelu2(self.disc_fc2(d1))
        d3 = self.disc_sig(self.disc_fc3(d2))

        return d3.view(-1, 1).squeeze(1)
