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
from gan_utils import *
from alignment_utils_mnist import *
import wandb
from constellations_tools_mnist import *
from contour import *
from mnist_classify import *

print(pygad.__version__)

# define trails parameters

trial_name = 'trial37'


out_path = 'Trials/'+trial_name+'/'
if(not path.exists(out_path)):
    os.mkdir(out_path)   
    

input_folder = '../Mnist-Constellations/Constellations/'

select_folder = '../Mnist-Constellations/Selected_Mnist/'
sel_files = os.listdir(select_folder)

input_files = os.listdir(input_folder)


mnist_cls = mnist_classifier()


g_obj = mnist_gan()


align_obj = alignment()
contour_obj = evaluate_contour()

pop_size = 1000

def initial_population(gan_obj,num_samples):
    samples = gan_obj.generate_latents(num_samples)
    return np.squeeze(samples.detach().cpu().numpy())

def fitness_func(ga_object,solutions, solution_idx):
    solutions= np.reshape(solutions,(len(solutions),init_pop.shape[1],1,1))
    solutions = torch.from_numpy(solutions)
    images = g_obj.generate_samples(solutions)
    dists = []
    for image in images:
        image = cv2.resize(image, (160,160))
        line = outline(image)
        #count = len(points_on_image(line,dots,3))
        dist= contour_obj.get_contour_score(line,dots,3)
        dists.append(dist)
    return dists

# save best solution at end of each generation

def print_best(ga_object):
    solution, solution_fitness, solution_idx= ga_object.best_solution()
    solution= np.reshape(solution,(1,init_pop.shape[1],1,1))
    solution = torch.from_numpy(solution)
    image_init = g_obj.generate_samples(solution)
    image = cv2.resize(image_init, (160,160))
    line = outline(image)
    edge = drawing_figure(line,base_img)
    count = len(points_on_image(line,dots,3))
    dist = contour_obj.get_contour_score(line,dots,2)
    #global gen_completed
    gen_completed = ga_object.generations_completed
    plt.imsave(out_path+'image_'+str(i)+'_'+str(ga_object.generations_completed)+'.jpg',image,cmap='gray')
    plt.imsave(out_path+'edge_'+str(i)+'_'+str(ga_object.generations_completed)+'.jpg',edge,cmap='gray')
    if(gen_completed == config_dict['num_generations']):
        global digit
        digit = mnist_cls.predict(image_init)
        print(img,'counts: ',count,'dists:',dist)
        #plt.imsave(out_path+'image'+file+str(ga_object.generations_completed)+'.jpg',image)
        #plt.imsave(out_path+'edge'+file+str(ga_object.generations_completed)+'.jpg',edge)
    return

# defining and running the ga search instance

config_dict = {
  "population_size": pop_size,
  "num_generations": 30,
  "num_parents_mating": 200,
  "num_genes": 100, 
  "crossover_type" : 'uniform',
  "mutation_type" : 'random',
  "parent_selection_type": "sss",
  "mutation_probability" : 0.5,
  "crossover_probability" : 0.01,
  "gene_type" : float
}

wandb.init(project="mnist_constellations_ai_search", entity="tarunkhajuria",config=config_dict)


correct_label = []
prediction = []
i =0
for file in input_files:
    img = input_folder+file
    if(file not in sel_files):
        continue
    try:
        base_img = np.array(Image.open(img))
        print(img)
    except:
        continue
    dots = stimuli_dots(base_img)
    init_pop = initial_population(g_obj,pop_size)
    print(init_pop.shape)
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
                       fitness_batch_size = 100,
                       on_generation= print_best)
    ga_instance.run()
    correct_label.append(int(file.split('_')[3][0]))
    prediction.append(digit)
    print(i,correct_label[i],prediction[i])
    i+=1

correct_label = np.array(correct_label)
prediction = np.array(prediction)
print('Accuracy:',np.sum(correct_label==prediction)/len(correct_label))








