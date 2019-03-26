from __future__ import print_function
import argparse
import os
import random
import numpy as np 
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

from models import DCGAN, GAN, VAE, DCVAE, DCGAN
from tools.model_train import train_GAN, train_VAE


def weights_init(network):
    '''
    Function to initialize the weights of the Networks.

    Input:
        network : generator or discriminator respectively
    '''
    classname = network.__class__.__name__
    if classname.find('Conv') != -1:
        network.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        network.weight.data.normal_(1.0, 0.02)
        network.bias.data.fill_(0)
    # elif classname.find('Linear') != -1:
    #     torch.nn.init.xavier_uniform_(network.weight)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='location of the dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='the height/width of the input image')

    parser.add_argument('--network', required=True, help='Type of network: GAN, DCGAN, VAE, DCVAE')
    parser.add_argument('--nc', type=int, default=3, help='no. of channels of the input image')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector (input of the Generator)')
    parser.add_argument('--ngf', type=int, default=64, help='no. of generator filters')
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate for the Generator')
    parser.add_argument('--ndf', type=int, default=64, help='no. of discriminator filters')
    parser.add_argument('--epochNum', type=int, default=25, help='no. of epochs')
    parser.add_argument('--learningRate', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--labelNoise', action='store_true', help='add random noise to the value of labels')
    parser.add_argument('--normalization', action='store_true', help='add batch normalization')
    parser.add_argument('--kld', action='store_true', help='criterion: KL divergence')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')

    parser.add_argument('--generator', default='', help="location of Generator model (to continue training)")
    parser.add_argument('--discriminator', default='', help="location of Discriminator model (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

    parser.add_argument('--manualSeed', type=int, help='manual seed')

    args = parser.parse_args()
    print(args)

    cuda = args.cuda
    cudnn.benchmark = True
    if cuda == True:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    num_epochs = args.epochNum   
    image_size = args.imageSize 
    nc = int(args.nc)
    batch_size = int(args.batchSize)

    nz = int(args.nz)
    ngf = int(args.ngf)
    ndf = int(args.ndf)

    dropout_rate = args.dropout
    learning_rate = args.learningRate
    beta1 = args.beta1
    
    normalization = args.normalization
    kld = args.kld
    label_noise = args.labelNoise
    output_folder = args.outf
    

    # Create Output Folder if it doesn't exist already
    try:
        os.makedirs(args.outf)
    except OSError:
        pass


    # Set manual / random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    
    # Load data. Make sure to abide by the rules of ImageFolder!
    dataset = datasets.ImageFolder(root=args.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                   )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                            shuffle=True, num_workers=int(args.workers))

    
    # Initializes Networks
    network = args.network
    if network == 'gan':
        generator = GAN.Generator(nz=nz, nc=nc, image_size=image_size, normalization=normalization).to(device)
        discriminator = GAN.Discriminator(nz=nz, nc=nc, image_size=image_size).to(device)
    
    elif network == 'dcgan':
        generator = DCGAN.Generator(nz=nz, nc=nc, ngf=ngf, image_size=image_size, 
                                    dropout_rate=dropout_rate, normalization=normalization).to(device)
        discriminator = DCGAN.Discriminator(nz=nz, nc=nc, ndf=ndf, image_size=image_size, 
                                            normalization=normalization).to(device)
        generator.apply(weights_init)
        discriminator.apply(weights_init)
    
    elif network == 'vae':
        encoder = VAE.Encoder(nc=nc, nz=nz, image_size=image_size, cuda=cuda, normalization=normalization).to(device)
        decoder = VAE.Decoder(nc=nc, nz=nz, image_size=image_size, normalization=normalization).to(device)
        
    elif network == 'dcvae':
        encoder = DCVAE.Encoder(nc=nc, nz=nz, ndf=ndf, image_size=image_size, cuda=cuda, normalization=normalization).to(device)
        decoder = DCVAE.Decoder(nc=nc, nz=nz, ngf=ngf, image_size=image_size, normalization=normalization).to(device)
        #encoder.apply(weights_init)
        #decoder.apply(weights_init)


    if network == 'gan' or network == 'dcgan':
        # Loads previously saved models (if chosen)
        if args.generator != '':
            generator.load_state_dict(torch.load(args.generator))
        print(generator)
        if args.discriminator != '':
            discriminator.load_state_dict(torch.load(args.discriminator))
        print(discriminator)

        # Optimizer initialization for the Discriminator & Generator
        optimizer_generator = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
        ## optimizerD = optim.SGD(discriminator.parameters(), lr=0.01)
        optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
        
        # Loss Function
        criterion = nn.BCELoss() # TODO: Try others too

        # Labels (1 for real images, 0 for generated images)
        real_label = 1
        fake_label = 0

        train_GAN(num_epochs=num_epochs, dataloader=dataloader, batch_size=batch_size, 
                  discriminator=discriminator, generator=generator, 
                  optimizer_discriminator=optimizer_discriminator, optimizer_generator=optimizer_generator,
                  label_noise=label_noise, output_folder=output_folder,
                  criterion=criterion, nz=nz, device=device)


    elif network == 'vae' or network == 'dcvae':
        # optimizer_encoder = optim.Adam(encoder.parameters(), lr=learning_rate, betas=(beta1, 0.999))
        # optimizer_decoder = optim.Adam(decoder.parameters(), lr=learning_rate, betas=(beta1, 0.999))

        optimizer_encoder = optim.SGD(encoder.parameters(), lr=learning_rate)
        optimizer_decoder = optim.SGD(decoder.parameters(), lr=learning_rate)

        criterion = nn.BCELoss() # TODO: Try others too

        train_VAE(num_epochs=num_epochs, dataloader=dataloader, batch_size=batch_size,
                  nz=nz, nc=nc, image_size=image_size,
                  encoder=encoder, decoder=decoder,
                  criterion=criterion, 
                  optimizer_encoder=optimizer_encoder, optimizer_decoder=optimizer_decoder,
                  output_folder=output_folder, device=device, kld=kld)