'''Script to evaluate the GAN images'''

import torch
from model import *
import numpy as np
class fmnist_generator:
    def __init__(self):
        '''initialise'''
        self.image_channels = 1
        self.noise_channels = 256
        self.gen_features = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gen =  Generator(self.noise_channels, self.image_channels, self.gen_features)
        self.gen.load_state_dict(torch.load('../FashionMnist_DCGAN/tests/light-violet-44/generator',map_location= self.device))
        return 
    def generate_latents(self,no_samples):
        '''generate random noise vectors'''
        noise = torch.randn(no_samples, self.noise_channels, 1, 1)
        return noise
        
    def generate_samples(self,noise):
        '''samples images given the noise vectors'''
        return self.gen(noise)
