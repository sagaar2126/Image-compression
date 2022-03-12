import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
from pathlib import Path
from skimage.io import imread
import cv2
from skimage.metrics import structural_similarity as ssim
from scipy.stats import spearmanr
import lpips

plt.rcParams['figure.dpi'] = 500
file=loadmat('..\hw5hw5.mat')
path_ref='../hw5refimgs'
path_blur='../hw5gblur'

df=pd.read_csv('LP_IPS.txt')
df.columns=[1]


def calc_mse(ref,blur):
    ref_image=imread(path_ref+"/"+str(ref))
    blur_image= imread(path_blur+"/"+str(blur))
    gray_ref = cv2.cvtColor(ref_image,cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.cvtColor(blur_image,cv2.COLOR_BGR2GRAY)
    mse=np.sum(np.square(gray_ref-gray_blur))/(gray_blur.shape[0]*gray_blur.shape[1])
    return mse

def calc_ssim(ref,blur):
    ref_image=imread(path_ref+"/"+str(ref))
    blur_image= imread(path_blur+"/"+str(blur))
    return ssim(ref_image,blur_image,multichannel=True)
    

def calc_lpips(ref,blur):
    ref_image=lpips.load_image((path_ref+"/"+str(ref)))
    blur_image=lpips.load_image((path_blur+"/"+str(blur)))
    ref_image = lpips.im2tensor(ref_image)
    blur_image = lpips.im2tensor(blur_image)
    loss_fn_alex = lpips.LPIPS(net='alex')
    return loss_fn_alex.forward(ref_image,blur_image)
    
    

dmos = np.array(file['blur_dmos']).reshape(-1,1)

# Naming names in 'refnames_blur' as same as names of images present in hw5refimgs
ref_name = file['refnames_blur'].reshape(-1,1)
ref_name = np.array([i[0] for i in ref_name]).reshape(-1,1)
ref_name = np.array(['refimgs'+str(i[0]) for i in ref_name]).reshape(-1,1)
# print(ref_name)

# Creating array of names of images present in  gblur folder.
blur_name = np.array(['gblurimg'+str(i)+".bmp" for i in range(1,175)]).reshape(-1,1)
# print(blur_name)

original_present=np.array(file['blur_orgs']).reshape(-1,1)
obj=np.c_[ref_name,blur_name,original_present,dmos]
obj=np.delete(obj,np.where(obj[:,2]=='1')[0],0)

MSE=[calc_mse(obj[i,0],obj[i,1]) for i in range(len(obj))]
SSIM=[calc_ssim(obj[i,0],obj[i,1]) for i in range(len(obj))]
# LPIPS=[calc_lpips(obj[i,0],obj[i,1]) for i in range(len(obj))]


MSEvsDmos=spearmanr(np.array(MSE),np.array(obj[:,3],dtype=float))
SSIMvsDmos= spearmanr(np.array(SSIM),np.array(obj[:,3],dtype=float))

# Computation of LPIPS takes time , So i once run and saved it in text file. 


    
LPIPSvsDmos= spearmanr(np.array(df),np.array(obj[:,3],dtype=float))
print(LPIPSvsDmos)
print(MSEvsDmos)
print(SSIMvsDmos)





plt.figure(figsize=(7,7 ))
plt.scatter(np.linspace(1,145,145),np.array(MSE),dtype=float)
plt.xlabel("Index ",fontsize=16)
plt.ylabel("MSE ",fontsize=16)
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.grid()
plt.show()







fig, ax = plt.subplots()
ax.scatter(np.linspace(1,145,145),np.array(MSE)) 
ax.set_xlabel("Index")
ax.set_ylabel("MSE")
ax.legend()
ax.grid(True)
plt.show()    

fig, ax = plt.subplots()
ax.scatter(np.linspace(1,145,145),np.array(SSIM),color='red') 
ax.set_xlabel("Index")
ax.set_ylabel("SSIM")
ax.legend()
ax.grid(True)
plt.show() 


fig, ax = plt.subplots()
ax.scatter(np.array(MSE),np.array(SSIM),marker='+') 
ax.set_xlabel("MSE")
ax.set_ylabel("SSIM")
ax.legend()
ax.grid(True)
plt.show() 

fig, ax = plt.subplots()
ax.scatter(np.array(MSE),np.array(obj[:,3],dtype=float),marker='+') 
ax.set_xlabel("MSE")
ax.set_ylabel("Dmos")
ax.legend()
ax.grid(True)
plt.show()

fig, ax = plt.subplots()
ax.scatter(np.array(SSIM),np.array(obj[:,3],dtype=float),marker='+') 
ax.set_xlabel("SSIM")
ax.set_ylabel("Dmos")
ax.legend()
ax.grid(True)
plt.show()

fig, ax = plt.subplots()
ax.scatter(np.array(MSE),MSEvsDmos,marker='+') 
ax.set_xlabel("MSE")
ax.set_ylabel("SSIM")
ax.legend()
ax.grid(True)
plt.show()

fig, ax = plt.subplots()
ax.scatter(np.array(SSIM),SSIMvsDmos,marker='+') 
ax.set_xlabel("MSE")
ax.set_ylabel("SSIM")
ax.legend()
ax.grid(True)
plt.show()




fig, ax = plt.subplots()
ax.scatter(np.array(SSIM).reshape(-1,1),np.array(obj[:,3],dtype=float), label='MSE vs Dmos ') 
ax.xlabel("MSE")
ax.ylabel("MOS")
ax.legend()
ax.grid(True)


fig, ax = plt.subplots()
ax.scatter(np.array(df),np.array(obj[:,3],dtype=float), label='LPIPS vs Dmos ')
ax.set_xlabel("LPIPS")
ax.set_ylabel("Dmos") 
ax.legend()
ax.grid(True)
plt.show()    
 


