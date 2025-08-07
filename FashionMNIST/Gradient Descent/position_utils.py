import cv2
import numpy as np


# find number of dots in an image that has only dots
def stimuli_dots(image): #counts gives the dots on the image
    ''' get the position of dots on the ground truth (dotted) or constellation image'''
    gray = cv2.resize(image,(160,160))
    ## threshold
    
    th, threshed = cv2.threshold(gray, 100, 255,cv2.THRESH_BINARY)
    ## findcontours
    #threshed = cv2.cvtColor(threshed, cv2.COLOR_BGR2GRAY)
    cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    return cnts

def find_dot_centres(cnts):
    '''returns the Xs,Ys i.e a list of x and y co-ordinates for centre of the given list of contours'''
    Xs,Ys= np.zeros(len(cnts)),np.zeros(len(cnts))
    for ind,cnt in enumerate(cnts):
        M = cv2.moments(cnt)
        Xs[ind] = int(M["m10"] / M["m00"])
        Ys[ind] = int(M["m01"] / M["m00"])
    return Xs,Ys
def drawing_figure(image,base=None):
    ''' find the drawn contour on the image'''
    cnts = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]
    if(base is None):
        draw_img = np.zeros((160,160)).astype(np.uint8)
    else:
        draw_img = base.astype(np.uint8)
    for s in cnts:
        for pt in s:
            draw_img[pt[0][1],pt[0][0]] =255
    return draw_img

def points_on_image(d_img,dots,lev):
    '''Gives you the points on the image that touch or are near the contour < lev pixels)'''
    count = 0
    sel_dots = []
    sel_ind = []
    xs,ys = find_dot_centres(dots)
    for ind,dot in enumerate(dots):
        radius = 0
        x,y,radius = xs[ind],ys[ind],0
        x1= int(max(x-radius-lev,0))
        x2 = int(min(x+radius+lev,d_img.shape[1]))
        y1= int(max(y-radius-lev,0))
        y2 = int(min(y+radius+lev,d_img.shape[0]))    
        if(np.max(d_img[y1:y2,x1:x2])>10):
            sel_dots.append(dot)
            sel_ind.append(ind)
    return sel_ind,sel_dots

def get_contours(edges,width=160, height=160):
    gray = cv2.resize(edges,(width,height))
    ## threshold
    #th, threshed = cv2.threshold(gray, 10, 255,cv2.THRESH_BINARY)
    ## findcontours
    cnts = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]
    sc =[]
    for ind,c in enumerate(cnts):
        area = cv2.contourArea(c)
        sc.append(c.copy())
    return sc