import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pylab
import numpy as np
from dcgan import Discriminator, Generator

class mnist_gan:
    def __init__(self):
        '''load weights and initialise the generator and discriminator'''
        
        num_gpu = 1 if torch.cuda.is_available() else 0
        # load the models
        #D = Discriminator(ngpu=num_gpu).eval()
        G = Generator(ngpu=num_gpu).eval()
        
        # load weights
        #D.load_state_dict(torch.load('GAN Weights/netD_epoch_99.pth'))
        G.load_state_dict(torch.load('GAN Weights/netG_epoch_99.pth'))
        
        if torch.cuda.is_available():
            #self.D = D.cuda()
            self.G = G.cuda()
        return
            
    def generate_latents(self,num_samples):
        latent_size = 100
        noise = torch.randn(num_samples, latent_size, 1, 1)
        return noise
    
    def generate_samples(self,vectors):
        if torch.cuda.is_available():
            vectors = vectors.cuda().to(torch.float)
        images = self.G(vectors).cpu().detach().numpy()
        images = np.squeeze(images)
        images = np.uint8((images+1)*255/2) #convert to uint8 scale
        return images
    
