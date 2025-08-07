import numpy as np
from PIL import Image, ImageDraw
import cv2

def outline(image):
    '''Function to generate outline using original image and segmentation mask
    Customise this function to generate differnt outlines and using different segmentation masks
    Returns: outline image'''
    image = np.uint8(cv2.blur(image, (3,3)))
    edges = cv2.Canny(image,40,120)
    return edges

'''Functions to generate constellation images from outlines'''
def draw_circle_white(draw,c,dist):
    r = dist
    shape = [(c[0]-r,c[1]-r),(c[0]+r,c[1]+r)]
    draw.ellipse(shape,fill=250) 
def draw_circle_black(draw,c,dist):
    r = dist
    shape = [(c[0]-r,c[1]-r),(c[0]+r,c[1]+r)]
    draw.ellipse(shape,fill=0) 

'''Function to generate a dotted image from an outline'''    
def generate_image_dotted(edges,dist=40,dot= 2):
    '''dist (d) gives the distance between dots, dot is the radius of each dot (r)'''
    im = Image.fromarray(edges)
    # create rectangle image 
    draw = ImageDraw.Draw(im)   
    img_shape = edges.shape
    px = im.load()
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if(px[j,i]==255):
                draw_circle_black(draw,(j,i),dist)
                px[j,i]=255
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if(px[j,i]==255):
                draw_circle_white(draw,(j,i),dot)
    return im
def add_noise(im,prob = 0.0001,dot =2 ):
    '''Function to add noise with a particular value of probability, Prob (p)'''
    draw = ImageDraw.Draw(im)
    img_shape = np.array(im).shape
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if(np.random.random()<prob):
                draw_circle_white(draw,(j,i),dot)
    return np.array(im)


def generate_constellations(image):
    ''' generate constellation image from normal mnist'''
    instance = cv2.resize(image, (160,160))
    line = outline(instance)
    dotted = generate_image_dotted(line,dist=13,dot= 1)
    const = add_noise(dotted,prob = 0.002,dot =1 )
    return const
    
    