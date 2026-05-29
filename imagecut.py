# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 11:11:32 2019

@author: brigc
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio


#%%
def Microscope_Cut(path_image,th_size=2): 
    # image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
    image = cv2.imread(path_image)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur=cv2.GaussianBlur(img_gray, (5, 5), sigmaX=10)
    # th, _ = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)  
    img_adaptive = cv2.adaptiveThreshold(img_blur,175,cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV,25,3)
    # img_edge = cv2.Canny(img_gray,th/15,th/5, apertureSize=3, L2gradient=True)
    # img_edge = edge_extend(img_edge, 20,iteration=5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_ex = cv2.morphologyEx(img_adaptive, cv2.MORPH_CLOSE, kernel,iterations=5) 
    cnts, _ = cv2.findContours(img_ex, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c_max = []
    # th_size = 5
    for i in range(len(cnts)):
        cnt = cnts[i]
        area = cv2.contourArea(cnt)
        if(area < np.pi*(th_size/2*3.48)**2):# or (area > 0.2*image.shape[0]*image.shape[1]):
            continue
        _,_,w,h = cv2.boundingRect(cnt)
        if(w*h > 0.8*image.shape[0]*image.shape[1]):
            continue        
        c_max.append(cnt)

    img_list=[]

    for cnt in c_max:
        x, y, w, h = cv2.boundingRect(cnt)
        imgOut=image[y:y+h,x:x+h,:]
        img_list.append(imgOut)
 
    return img_list#, image_coi


#%%
# def Flowcam_Cut(tif_path,target_dir,filename):
def Flowcam_Cut(tif_path):
    #获取图片
    path=tif_path
    img=cv2.imread(path)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, 1),0)
    (_, thresh) = cv2.threshold(blurred, 1, 255, cv2.THRESH_BINARY)
    cnts,_ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    imgs = []
    for i in range(len(cnts)):
        cnt=np.squeeze(cnts[i])
#        print(cnt)
        Xs = cnt[:,0]
        Ys = cnt[:,1]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        height = y2 - y1
        width = x2 - x1
#        print(x1,x2,y1,y2)
        crop_img = img[y1:y1+height, x1:x1+width]
        imgs.append(crop_img)

    return imgs

