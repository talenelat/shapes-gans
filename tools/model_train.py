'''
Module to train the models, plot losses and generate GIF animations.
'''

from __future__ import print_function
import random
import numpy as np 
import math
import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import imageio
import os


def gen_animated(output_folder):
    root_jpg = f'{output_folder}'
    images = []

    for file_name in os.listdir(root_jpg):
        if file_name.endswith('.png'):
            images.append(imageio.imread(os.path.join(root_jpg, file_name)))
    imageio.mimsave(f'{output_folder}/evol.gif', images)


def plot_losses_gan(gen_losses, disc_losses, output_folder):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(gen_losses,label="Generator")
    plt.plot(disc_losses,label="Discriminator")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{output_folder}/loss.jpg')

def plot_losses_vae(loss, output_folder):
    plt.figure(figsize=(10,5))
    plt.title("VAE Loss During Training")
    plt.plot(loss,label="VAE")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{output_folder}/loss.jpg')


def train_GAN(num_epochs, dataloader, batch_size,
              discriminator, generator, 
              optimizer_discriminator, optimizer_generator,
              criterion, nz, device, label_noise, output_folder):

    # Fixed Latent Input to check evolution in time
    fixed_noise = torch.randn(1, nz, 1, 1, device=device)

    # Labels for the real images (=1) and generated/'fake' images (=0)
    real_label = 1
    fake_label = 0

    gen_losses = []
    disc_losses = []
    
    

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 1):
            
            discriminator.train()
            generator.train()

            ### Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))

            # Set gradients of the Discriminator to 0 
            optimizer_discriminator.zero_grad()

            # Load current batch of images
            real_images = data[0].to(device)
            batch_size = real_images.size(0)

            # Add noise to the label
            if label_noise == True:
                label = torch.full((batch_size,), real_label + np.random.uniform(-0.1, 0.2), device=device)
            elif label_noise == False:
                label = torch.full((batch_size,), real_label, device=device)

            # Evaluate Discriminator on real images
            prediction_real = discriminator(real_images)
            error_disc_real = criterion(prediction_real, label)
            error_disc_real.backward()
            D_x = prediction_real.mean().item()

            # Train with generated images
            z = torch.randn(batch_size, nz, 1, 1, device=device)

            # Generate fake images
            fake_images = generator(z)
            
            # Label fake images with 0 (+ noise)
            if label_noise == True:
                label.fill_(fake_label + np.random.uniform(0, 0.2))
            elif label_noise == False:
                label.fill_(fake_label)

            # Evaluate Discriminator on generated images
            prediction_fake = discriminator(fake_images.detach())
            error_disc_fake = criterion(prediction_fake, label)
            error_disc_fake.backward()

            # Prediction on generated images before Discriminator update
            D_G_z1 = prediction_fake.mean().item()
            error_disc = error_disc_real + error_disc_fake

            # Update Discriminator
            optimizer_discriminator.step()

            ### Train Generator: maximize log(D(G(z)))

            optimizer_generator.zero_grad()

            if label_noise == True:
                label.fill_(real_label + np.random.uniform(0, 0.2))
            elif label_noise == False:
                label.fill_(real_label)

            # Prediction on generated images after Discriminator update
            prediction_fake_updated = discriminator(fake_images)
            error_gen = criterion(prediction_fake_updated, label)
            error_gen.backward()
            D_G_z2 = prediction_fake_updated.mean().item()

            # Update Generator
            optimizer_generator.step()

            gen_losses.append(error_gen.item())
            disc_losses.append(error_disc.item())

            # Print status
            print('[Epoch %d of %d][Batch %d of %d] -> Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch+1, num_epochs, i, len(dataloader), error_disc.item(), error_gen.item(), D_x, D_G_z1, D_G_z2))

            # Save batch of fake images (fixed noise to observe evolution over time)
            if i % 10 == 0:
                generator.eval()
                fake_fixed = generator(fixed_noise)
                vutils.save_image(fake_fixed.detach(),
                                  '%s/gen_samples_epoch_%03d_%03d.png' % (output_folder, epoch+1, i),
                                  normalize=True)

        # Save Models to Output Folder
        torch.save(generator.state_dict(), '%s/generator_epoch_%d.pth' % (output_folder, epoch+1))
        torch.save(discriminator.state_dict(), '%s/discriminator_epoch_%d.pth' % (output_folder, epoch+1))
    plot_losses_gan(gen_losses=gen_losses, disc_losses=disc_losses, output_folder=output_folder)
    gen_animated(output_folder=output_folder)


def train_VAE(num_epochs, dataloader, batch_size,
              nz, nc, image_size,  
              encoder, decoder,
              optimizer_encoder, optimizer_decoder,
              criterion, output_folder, device, kld=False):

    # Fixed Latent Input to check evolution in time
    fixed_noise = torch.randn(batch_size, nz, device=device)

    train_loss = 0
    losses = []

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 1):
            encoder.train()
            decoder.train()

            # Set the gradients of the Encoder and Decoder to 0
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()

            # Load current batch of images
            real_images = data[0].to(device)
            batch_size = real_images.size(0)

            # Encode real images using the Encoder
            z, mu, logvar = encoder(real_images)

            # Generate images using the Decoder
            fake_images = decoder(z)
            
            # Evaluate the performance of the system by comparing
            # the generated/'fake' images with the real ones
            if kld == True:
                loss = loss_function(fake_images, real_images, mu, logvar)
            else:
                loss = criterion(fake_images, real_images)
            loss.backward()
            losses.append(loss.item())
            train_loss += loss.item()

            # Update the Encoder and the Decoder
            optimizer_encoder.step()
            optimizer_decoder.step()

            print('[Epoch %d of %d][Batch %d of %d] -> Loss: %.4f'
                  % (epoch+1, num_epochs, i, len(dataloader), loss.item()))
            if i % 10 == 0 and i != len(dataloader):
                decoder.eval()
                fake_fixed = decoder(fixed_noise)
                vutils.save_image(fake_fixed.view(batch_size, nc, image_size, image_size),
                                  '%s/gen_samples_epoch_%03d_%03d.png' 
                                  % (output_folder, epoch+1, i+1),
                                  normalize=True)

        # Save Models to Output Folder
        torch.save(encoder.state_dict(), '%s/encoder_epoch_%d.pth' % (output_folder, epoch+1))
        torch.save(decoder.state_dict(), '%s/decoder_epoch_%d.pth' % (output_folder, epoch+1))
    gen_animated(output_folder=output_folder)
    plot_losses_vae(loss=losses, output_folder=output_folder)


def loss_function(fake_images, x, mu, logvar):
    reconstruction_function = nn.BCELoss()
    reconstruction_function.size_average = False

    scaling_factor = fake_images.shape[0]*fake_images.shape[1]*fake_images.shape[2]*fake_images.shape[3]
    
    BCE = reconstruction_function(fake_images, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    KLD /= scaling_factor

    return BCE + KLD