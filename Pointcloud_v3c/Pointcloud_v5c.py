import random
import pandas as pd
import numpy as np
import time
import math

from PIL import Image

import argparse

import matplotlib.pyplot as plt
import operator

from netCDF4 import Dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from scipy.optimize import minimize
import scipy.stats

def m_poly(xs,zs,deg):

    
   
            
    xs = xs[:, np.newaxis]
    zs = zs[:, np.newaxis]
    model = LinearRegression() 
    model.fit(xs, zs)
    y_pred = model.predict(xs) 

   
    #Polynomial regression
    polynomial_features= PolynomialFeatures(degree=deg) 
    x_poly = polynomial_features.fit_transform(xs)
    model = LinearRegression() 
    model.fit(x_poly, zs) 
    y_poly_pred = model.predict(x_poly) 
    y_poly_pred = np.concatenate(y_poly_pred, axis=None)
    zs = np.concatenate(zs, axis=None)
    xs = np.concatenate(xs, axis=None)
    
    #Polyfit to retreive coefficient
    z = np.polyfit(xs, zs, deg)
    
    #Function returning the polynomial interpolation
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
        res = minimize(obj,xs[0],method='COBYLA')
        dist.append(res['fun'])

    return dist




img = Image.open("Processed image from ERS4.jpg_.png")
img_long = img.size[1]
#4 Corners from V-optics picture reference x,-y
x1 = np.array([120,2448,2448,24])
y1 = np.array([180,248,2104,2008])

#transformation standard x,y
y1 = img_long-y1

#tranformation to D point reference
xd = x1[3]
yd = y1[3]
x1 = x1 - xd
y1 = y1 - yd


#angle computation
teta1 = abs(y1[2])/x1[2]
teta1 = math.atan(teta1)


A1 = [math.cos(teta1),-math.sin(teta1)]

teta2 = abs(x1[0])/y1[0]
teta2 = math.atan(teta2)

A1 = [math.cos(teta1),-math.sin(teta2)]
A2 = [math.sin(teta1),math.cos(teta2)]

#Rotation Matrix
rot =[]
rot.append(A1)
rot.append(A2)
mat_rot = np.array(rot)

#ABCD Points from V_Optics
mat1 = []
mat1.append(x1)
mat1.append(y1)
B = np.array(mat1)

#ABCD after rotation
mat_ABCD = mat_rot.dot(B)

#Recovering OpenCV result data
monFichier = open("./Deflectometrie_OpenCV/result.txt" , "r") 
temp= monFichier.read()
temp = temp.split("\n")#Each row is a cell

m_def = []
xUL = []
xDR = []
yUL = []
yDR = []

for i in range(0,len(temp)-1):
    line = temp[i].split(",")
    m_def.append(line[0])
    xUL.append(int(line[1]))
    xDR.append(int(line[2]))
    yUL.append(int(line[3]))
    yDR.append(int(line[4]))

#Transformation of defaults areas to D reference
yUL = np.array(yUL)
yUL2 = yUL
yUL = img_long - yUL
yDR = np.array(yDR)
yDR2 = yDR
yDR = img_long - yDR
xUL = np.array(xUL)
xUL2 = xUL
xDR = np.array(xDR)
xDR2 = xDR
xUL = xUL - xd
yUL = yUL - yd
xDR = xDR - xd
yDR = yDR - yd

#Building UL matrix point
UL = []
UL.append(xUL)
UL.append(yUL)
mat_UL = np.array(UL)


#Building DR matrix point
DR = []
DR.append(xDR)
DR.append(yDR)
mat_DR = np.array(DR)

#UL & DR rotated matrix
mat_UL_rot = mat_rot.dot(mat_UL)
mat_DR_rot = mat_rot.dot(mat_DR)


#Zivid data
img2 = Image.open("shot3.jpg")

img2 = img2.rotate(180)

img2_long = img2.size[1]

#4 Corners from Zivid picture reference x,-y
x2 = np.array([690,1262,1262,654])
y2 = np.array([108,176,728,738])
x2r = x2
y2r = y2

#transformation standard x,y
y2 = img2_long-y2

#transformation to D point reference
xd2 = x2[3]
yd2 = y2[3]
x2 = x2 - xd2
y2 = y2 - yd2


#angle computation
teta1 = abs(y2[2])/x2[2]
teta1 = math.atan(teta1)
if y2[2]>y2[3]:
    teta1 = -teta1
A1 = [math.cos(teta1),-math.sin(teta1)]


teta2 = abs(x2[0])/y2[0]
teta2 = math.atan(teta2)
if x2[1]<x2[0]:
    teta2 = -teta2


A1 = [math.cos(teta1),-math.sin(teta2)]
A2 = [math.sin(teta1),math.cos(teta2)]

#Rotation Matrix
rot2 =[]
rot2.append(A1)
rot2.append(A2)
mat_rot2 = np.array(rot2)

#ABCD Points from V_Optics
mat2 = []
mat2.append(x2)
mat2.append(y2)
B2 = np.array(mat2)

#ABCD2 after rotation
mat_ABCD2 = mat_rot2.dot(B2)


##Length DC in V_Optics referential
DC1 = mat_ABCD[0][2]-mat_ABCD[0][3]
##Length DC in V_Optics referential
DC2 = mat_ABCD2[0][2]-mat_ABCD2[0][3]
##Length DA in V_Optics referential
DA1 = mat_ABCD[1][0]-mat_ABCD[1][3]
##Length DC in V_Optics referential
DA2 = mat_ABCD2[1][0]-mat_ABCD2[1][3]

##Transformation scale from V-optics to Zivid
k = DC2/DC1
l = DA2/DA1

##Transformation from V-Optics to Zivid scale
x3_UL = mat_UL_rot[0]*k
y3_UL = mat_UL_rot[1]*l 
x3_DR = mat_DR_rot[0]*k 
y3_DR = mat_DR_rot[1]*l

##Point Matrix to prepare rotations of all points back
#Building UL matrix point
UL2 = []
UL2.append(x3_UL)
UL2.append(y3_UL)
mat_UL2 = np.array(UL2)


#Building DR matrix point
DR2 = []
DR2.append(x3_DR)
DR2.append(y3_DR)
mat_DR2 = np.array(DR2)

#----------------------------------------------
#Rotation Matrix
A1 = [math.cos(-teta1),-math.sin(-teta2)]
A2 = [math.sin(-teta1),math.cos(-teta2)]

rot3 =[]
rot3.append(A1)
rot3.append(A2)
mat_rot3 = np.array(rot3)

#UL & DR rotated matrix
mat_UL_rot2 = mat_rot3.dot(mat_UL2)
mat_DR_rot2 = mat_rot3.dot(mat_DR2)


#Transformation from D reference to standard xy
x3_UL = mat_UL_rot2[0] + xd2
y3_UL = mat_UL_rot2[1] + yd2
x3_DR = mat_DR_rot2[0] + xd2
y3_DR = mat_DR_rot2[1] + yd2


#Transformation in Zivid pix to x,-y reference
y3_UL = img2_long - y3_UL
y3_DR = img2_long - y3_DR



##offset creation to locate squares
#offsetl = 20
#offsetv = 20
#
##Storing data in txt file of all squares transformed in Zivid jpeg
#m_file = open("result_zivid.txt" , "w") 
#
##Display the default area on Zivid JPG file
#for j in range(0,len(m_def)):
#    
#    xul = int(x3_UL[j]-offsetl)
#    yul = int(y3_UL[j])
#    xdr = int(x3_DR[j]+offsetl)
#    ydr = int(y3_DR[j])
#    
#    m_file.write(m_def[j]+","+str(xul)+","+str(yul)+","+str(xdr)+","+str(ydr)+",\n")
#    
#    
#    for i in range(xul,xdr,1) :
#       img2.putpixel((i,yul),(255,0,0))
#       img2.putpixel((i,yul+1),(255,0,0))
#     
#    for i in range(xul,xdr,1) :
#       img2.putpixel((i,ydr),(255,0,0))
#       img2.putpixel((i,ydr+1),(255,0,0))
#       
#    for i in range(yul,ydr,1) :
#       img2.putpixel((xul,i),(255,0,0))
#       img2.putpixel((xul+1,i),(255,0,0))  
#       
#    for i in range(yul,ydr,1) :
#       img2.putpixel((xdr,i),(255,0,0))
#       img2.putpixel((xdr+1,i),(255,0,0))
#m_file.close()
#plt.subplot(111)
#plt.imshow(img2)
#plt.show()
#
#
#Recover z value for the areas
zdf_Filename = "shot3.zdf"
data = Dataset(zdf_Filename)
xyz = data['data']['pointcloud'][:,:,:]
data.close()

a = 0
m_step = int(1)


xUL2 = x3_UL
xDR2 = x3_DR

yUL2 = y3_UL
yDR2 = y3_DR

x_len = len(xyz[0])
y_len = len(xyz)

#Roation matrix to 180°
xyz = np.rot90(xyz, 2)


xs = []
ys = []
zs = []

altitude = []

print(min(y2r))
print(max(y2r))
print(min(x2r))
print(max(x2r))


for i in range(min(y2r),max(y2r),m_step):
    
    temp = []#line of altitude
    temp2 = []
    
    for j in range(min(x2r),max(x2r),m_step):
        
        xs.append(j)
        ys.append(i)
        zs.append(round(xyz[i][j][2],2))
        temp.append(round(xyz[i][j][2],2))

   
    #Replacing the nan values by the median of the list to reduce the impact
    temp = np.array(temp)
    temp2 = temp[~np.isnan(temp[:])]
    m_median = np.median(temp2)
    
    for a in range(0,len(temp)):
        if str(temp[a])=="nan":
            temp[a] = m_median
    
    altitude.append(temp)
    
##Storing data in txt file of all squares transformed in Zivid jpeg
m_file = open("mesure_zivid.txt" , "w") 

distance = []

for i in altitude:
    temp3 = []
    temp3 = m_poly(np.arange(min(x2r),max(x2r)),np.array(i),3)
    distance.append(temp3)
    m_file.write(str(temp3)+"\n")

m_file.close()





#plt.plot(reg[0])
#plt.plot(result[0])

#plt.show()
    
###zs3 to be used only for 3D graph 
#zs3 = reg
#for val in result:
#    for uni in val:
#        zs3.append(uni)
#        
#zs3 = np.array(zs3)
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#ax.scatter(xs, ys, zs3, marker='.',edgecolor = 'none')
#
#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')
#
#
#plt.show()

    