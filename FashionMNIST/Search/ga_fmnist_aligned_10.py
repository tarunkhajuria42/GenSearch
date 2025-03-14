#!/usr/bin/env python
# coding: utf-8

import pygad
import numpy as np
from importlib import reload
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import torch
import sys
from position_utils import *
from os import path
import os
import cv2
from generator import *
#from alignment_utils_mnist import *
import wandb
#from constellations_tools_mnist import *
from contour import *
from classifier import *

# define trails parameters


run = wandb.init()

correct_count =0 

trial_name = run.name
print(trial_name)
out_path = 'Trials/'+trial_name+'/'
if(not path.exists(out_path)):
    os.mkdir(out_path)   

# test image

test_inputs = np.load('../../Fashion Mnist/dataset/fashion_constellation_test_sel_9.npz')
for ind in range(0,10):
    base_img = test_inputs['images'][ind].astype(np.uint8)
    label = test_inputs['labels'][ind]
    
    mnist_cls = fmnist_classifier()
    dots = stimuli_dots(base_img)
    print('total dots',len(dots))
    print('label', label)
    
    g_obj = generator() 
     
    # remove bottom up lines
    
    contour_obj = evaluate_contour()
    
    pop_size = 500
    
    gen_completed = 0
    
    def initial_population(gan_obj,num_samples):
        samples = gan_obj.generate_latents(num_samples)
        return np.squeeze(samples.detach().cpu().numpy())
    
    init_pop = initial_population(g_obj,pop_size)
    
    
    def fitness_func(ga,solution, solution_idx):
        solution= np.reshape(solution,(1,init_pop.shape[1],1,1))
        solution = torch.from_numpy(solution)
        image = g_obj.generate_samples(solution)
        image = cv2.resize(image, (160,160))
        line = outline(image)
        # plt.imsave(out_path+'image'+str(ga.generations_completed)+'_'+str(solution_idx)+'.jpg',image)
        # plt.imsave(out_path+'edge'+str(ga.generations_completed)+'_'+str(solution_idx)+'.jpg',line)
        #count = len(points_on_image(line,dots,2))  
        dists = contour_obj.get_contour_score(line,dots,2)
        #print(dists)
        return dists
    
    # save best solution at end of each generation
    
    def print_best(ga_object):
        print('best')
        solution, solution_fitness, solution_idx= ga_object.best_solution()
        solution= np.reshape(solution,(1,init_pop.shape[1],1,1))
        solution = torch.from_numpy(solution)
        image_init = g_obj.generate_samples(solution)
        image = cv2.resize(image_init, (160,160))
        line = outline(image)
        edge = drawing_figure(line,base_img)
        count = len(points_on_image(line,dots,3))
        dists = contour_obj.get_contour_score(line,dots,3)
        print('counts: ',count,' dists: ',dists)
        #global gen_completed
        gen_completed = ga_object.generations_completed
        plt.imsave(out_path+'image'+'_'+str(ind)+'_'+str(ga_object.generations_completed)+'.jpg',image)
        plt.imsave(out_path+'edge'+'_'+str(ind)+'_'+str(ga_object.generations_completed)+'.jpg',edge)
        if(gen_completed == config_dict['num_generations']):
            image_class = cv2.resize(image_init, (28,28))
            digit = mnist_cls.predict(image_class)
            print(label,digit)
            if(digit == label):
                global correct_count
                correct_count+=1
                print(correct_count)
        return
    
    # defining and running the ga search instance
    
    config_dict = {
      "population_size": init_pop.shape[0],
      "num_generations": 30,
      "num_parents_mating": 200,
      "num_genes": init_pop.shape[1], 
      "crossover_type" : 'uniform',
      "mutation_type" : 'random',
      "parent_selection_type": "sss",
      "mutation_probability" : 0.5,
      "crossover_probability" : 0.01,
      "gene_type" : float
    }
    
    wandb.init(project="mnist_constellations_ai_search", entity="tarunkhajuria",config=config_dict)
    
    ga_instance = pygad.GA(num_generations= config_dict['num_generations'],
                           num_parents_mating= config_dict['num_parents_mating'],
                           num_genes=config_dict['num_genes'], 
                           fitness_func= fitness_func,
                           crossover_type = config_dict['crossover_type'],
                           mutation_type = config_dict['mutation_type'],
                           parent_selection_type=config_dict['parent_selection_type'],
                           mutation_probability = config_dict['mutation_probability'],
                           crossover_probability = config_dict['crossover_probability'],
                           gene_type = config_dict['gene_type'],
                           initial_population = init_pop,
                           on_generation= print_best)
    
    ga_instance.run()
    ga_instance.save(filename=out_path+'model.ga')
    print(correct_count,ind)
    
    
    
    
    
    
    
