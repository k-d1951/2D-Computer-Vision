import random
import pandas as pd
import numpy as np
import time

from math import *
from PIL import Image

import argparse

import matplotlib.pyplot as plt
import operator

from netCDF4 import Dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from scipy.optimize import minimize
import scipy.stats

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())



lim_l = 660
lim_r = 1300

img = Image.open(str(args["image"])+".jpg")


tab_R = []
tab_G = []
tab_B = []

img_long = img.size[0]
imh_high = img.size[1]

#line to be analysed
pixline = int(660)

#horizontal
for i in range(0,int(img_long)) :
   img.putpixel((i,pixline),(255,0,0))
   img.putpixel((i,pixline+1),(255,0,0))
   #img.putpixel((i,pixline+2),(255,0,0))
   img.putpixel((i,pixline-1),(255,0,0))
   #img.putpixel((i,pixline-2),(255,0,0))
   

zdf_Filename = str(args["image"])+".zdf"

data = Dataset(zdf_Filename)
xyz = data['data']['pointcloud'][:,:,:]
data.close()

m_len = int(len(xyz[pixline]))
m_tabz = []



for i in range(lim_l,lim_r,1):
    m_tabz.append(round(xyz[pixline][i][2],2))
   

xs = np.arange(lim_l,lim_r,1)
zs = np.array(m_tabz)


##linear regression
#lr = scipy.stats.linregress(xs,zs)
#ys = np.array(lr[0]*xs+lr[1])
##
###difference to be aligned on x
#diff = np.array(zs-ys)
#
##sigma
#ec = np.std(diff)
##mean
#moy = np.mean(diff)
#
#
###Thresehold sigma
#ec = ec *1
#
###Highlight point out of extreams
#diff2 = []
#for i in diff:
#    if i <= -ec or i>=ec:
#        diff2.append(i)
#    else:
#        diff2.append(0)
##
###Remove the noise as standard
#diff3 = []
#for i in diff2:
#    if i <= -ec:
#        diff3.append(i+ec)
#    elif i>=ec :
#        diff3.append(i-ec)
#    else:
#        diff3.append(0)     

#Cleaning removing nan
zs2 = zs[~np.isnan(zs[:])]
mean = np.median(zs2)

for a in range(0,len(zs)):
    if str(zs[a])=="nan":
        zs[a] = mean
        

xs = xs[:, np.newaxis]
zs = zs[:, np.newaxis]

model = LinearRegression() 
model.fit(xs, zs)
y_pred = model.predict(xs) 

deg = 3

#Polynomial regression
polynomial_features= PolynomialFeatures(degree=deg) 
x_poly = polynomial_features.fit_transform(xs)

model = LinearRegression() 
model.fit(x_poly, zs) 
y_poly_pred = model.predict(x_poly) 


y_poly_pred = np.concatenate(y_poly_pred, axis=None)
zs = np.concatenate(zs, axis=None)
xs = np.concatenate(xs, axis=None)

diff = np.array(zs-y_poly_pred)

z = np.polyfit(xs, zs, deg)


#function returning the polynomial interpolation
def f(x,coef,deg):

    val = float(0)
    for i in range(0,deg+1):
       val = val + coef[i]*x**(deg-i)
    return(val)


#Function to be minimized : distance between point cloud and polynomial interpolation
def obj(x):
    result = np.sqrt(((x - m_pt[0])**2) + ((f(x,z,deg) - m_pt[1])**2))
    return(result)

#log all distances in a tab
dist = []
for i in range(0,len(xs)):
    m_pt = [int(xs[i]),zs[i]]
    res = minimize(obj,lim_l,method='COBYLA')
    dist.append(res['fun'])


plt.subplot(121)
#plt.plot(xs,zs)
#plt.plot(xs,y_pred)
#plt.plot(xs, y_poly_pred)
#plt.plot(xs,f(xs,z,deg))

#plt.plot(1107, f(1107,z,deg), 'bo', label='point')
#plt.plot([P[0], P2[0]], [P[1], P2[1]], 'b-', label='shortest distance')

#plt.plot(ys)
#plt.plot(xs,diff)
plt.plot(xs,dist)

#plt.title("Limited altitude for measures")
#plt.plot(diff3)
plt.grid()


#
plt.subplot(122)
##plt.plot(diff2)
##plt.title("Limited altitude")
##plt.hist(diff,bins=int(sqrt(len(ys))))
##plt.plot(diff3)
##plt.grid()
plt.imshow(img)
plt.show()   
