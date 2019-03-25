import torch
import torch.nn as nn
import torchvision


class Encoder(nn.Module):
    def __init__(self, nc, nz, image_size, cuda=False, normalization=False):
        super(Encoder, self).__init__()

        self.nc = nc
        self.nz = nz
        self.image_size = image_size
        self.cuda = cuda
        self.normalization = normalization

        # Input: (nc * image_size * image_size)
        
        self.enc_fc1 = nn.Linear(self.nc * self.image_size * self.image_size,
                                 self.nc * self.image_size * self.image_size // 2)
        self.enc_lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.enc_bn1 = nn.BatchNorm1d(self.nc * self.image_size * self.image_size // 2)
        # Output: (nc * image_size * image_size // 2)

        self.enc_fc2 = nn.Linear(self.nc * self.image_size * self.image_size // 2,
                                 self.nc * self.image_size * self.image_size // 4)
        self.enc_lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.enc_bn2 = nn.BatchNorm1d(self.nc * self.image_size * self.image_size // 4)
        # Output: (nc * image_size * image_size // 4)

        self.enc_fc_mu = nn.Linear(self.nc * self.image_size * self.image_size // 4, self.nz)
        self.enc_fc_log = nn.Linear(self.nc * self.image_size * self.image_size // 4, self.nz)
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
        image = image.view(-1, self.nc * self.image_size * self.image_size)
        if self.normalization == True:
            e1 = self.enc_bn1(self.enc_lrelu1(self.enc_fc1(image)))
            e2 = self.enc_bn2(self.enc_lrelu2(self.enc_fc2(e1)))
        else:
            e1 = self.enc_lrelu1(self.enc_fc1(image))
            e2 = self.enc_lrelu2(self.enc_fc2(e1))
        mu = self.enc_fc_mu(e2)
        logvar = self.enc_fc_log(e2)
        z = self.calculate_z(mu=mu, logvar=logvar)
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, nz, nc, image_size, normalization=False):
        super(Decoder, self).__init__()

        self.nc = nc
        self.nz = nz
        self.image_size = image_size
        self.normalization = normalization

        # Input: (nz)

        self.dec_fc1 = nn.Linear(self.nz, 
                                 self.nc * self.image_size * self.image_size // 2)
        self.dec_sig1 = nn.Sigmoid()
        self.dec_bn1 = nn.BatchNorm1d(self.nc * self.image_size * self.image_size // 2)
        # Output: (self.nc * self.image_size * self.image_size // 2)

        self.dec_fc2 = nn.Linear(self.nc * self.image_size * self.image_size // 2, 
                                 self.nc * self.image_size * self.image_size)
        self.dec_sig = nn.Sigmoid()
        # Output: (self.nc * self.image_size * self.image_size)

    def forward(self, z):
        if self.normalization == True:
            d1 = self.dec_bn1(self.dec_sig1(self.dec_fc1(z)))
        else:
            d1 = self.dec_sig1(self.dec_fc1(z))
        d2 = self.dec_sig(self.dec_fc2(d1))
        return d2