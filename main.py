import numpy as np
import matplotlib.pyplot as plt
import skimage.io 
from scipy.fftpack import fft, dct,idct
import os
import sys
import pandas as pd
from skimage.metrics import mean_squared_error as mse
import cv2
plt.rcParams['figure.dpi'] = 500


Path='..\cameraman.tif'

# Computing DCT
def ComputDCT(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')


#Computing Inverse DCT
def InverseDCT(img):
    return idct(idct(img.T, norm='ortho').T, norm='ortho')  



#Quantization
def Quantize(img,Q):
    # print(np.floor(np.divide(img,Q)+0.5))
    return np.floor(np.divide(img,Q)+0.5)

# Finding  class and indexes of DCT coefficient and . class is same as DC difference cateogory taught in class.

#    -1,1 ---> class 0     2^{0}+0=1         
#  -3,-2,2,3---->class 1   2^{1}+0=2,2^{1}+1=3 for -ve values abs value is taken , beacuse (-3,3) belongs to same class. Here (0,1) are the indexes
# -7,-6,-5,-4,4,5,6,7--->class 2   2^{2}+0=2,2^{2}+1=5,2^{2}+2=6,2^{2}+3=7
#  index varies from (0 to (2**(class)-1)) , Taken only absolute value for computing class and index. Class remains same for +ve and -ve value , indexes change

def Index(A): 
    if (abs(A)==0):
        pass
    else:                                # i class , j index
        for i in range(15):
            for j in range(2**i):
                B=(2**i)+j
                if(abs(int(A))==B):
                    return [i,j]


# generating all the functions of n bits recursivley. for n=2 {00,01,10,11} 
def func(n):
    def genbin(n, bs=''):
         if len(bs) == n:
              l.append(bs)
         else:
              genbin(n, bs + '0')
              genbin(n, bs + '1')
     
    l=[]
    genbin(n)     
    return l


def Convertbitstream(a,pixel):

    #----------------------
    if pixel==0:
        return '0'                                      #Generating prefixes like 110, 11110 if DCT coefficient is not zero
    
    bits='0'
    for i in range(a[0]+1):
        bits='1'+bits
    l=func(a[0]+1)                                 # l contains all the functions of n bits. eg. for n=2 {00,01,10,11} 
   #--------------------------
        
   
   #----------------------------------------------------------------------------------------------------------- 
    if(pixel>0):
       index1=int(((2**a[0]))+a[1])
                                           # Mapping back correct indexes i.e taking account for -ve DCT coefficients and 
       return bits+l[index1]               # selecting suffix using correct indexes
        
    if(pixel<0):
       # print(bits)
       index2=int(((2**a[0]))-1-a[1])
       return bits+l[index2]
            
   #----------------------------------------------------------------------------------------------------------------------       


# This function will encode each DCT coefficients in 8x8 vlock
def encode(arr):
    encoded=""
    for i in range(8):
        for j in range(8):
            a=Index(arr[i,j])
            C=Convertbitstream(a,arr[i,j])
            print((arr[i,j],C))
            encoded=encoded+C
    return encoded   
            

    
def problem1():
    
    image=skimage.io.imread(Path)
    h,w=image.shape
    A=[i*8 for i in range(33)]    
    B=[j*8 for j in range(33)]
    Q=[[20,10,10,10,40,40,40,40],
      [10,10,10,10,40,40,40,40],
      [10,10,10,10,40,40,40,40],
      [10,10,10,10,40,40,40,40],
      [40,40,40,40,40,40,40,40],
      [40,40,40,40,40,40,40,40],
      [40,40,40,40,40,40,40,40],
      [40,40,40,40,40,40,40,40]]
   
    Q=np.array(Q)
    k=1
    Reconstructed_image=np.zeros(image.shape)
    bitstream=""    
    
    for i in range(len(A)-1):
       l=1
       for j in range(len(B)-1):
          block=image[A[i]:A[k],B[j]:B[l]]                                                         
          DCT=ComputDCT(block)
          Reconstructed_image[A[i]:A[k],B[j]:B[l]]= InverseDCT((np.multiply(Quantize(DCT,Q),Q)))
          bitstream=bitstream+encode(Quantize(DCT,Q).astype(int))
          l+=1
       k+=1 
    
    plt.imshow(Reconstructed_image,cmap='gray')
    plt.show()
    
    MSE=mse(image,Reconstructed_image)
    print("mean squared error is:", MSE)   
    print("Size of output bitstream is :",str(len(bitstream))+"bits")   


def problem2():
    image=skimage.io.imread(Path)
    h,w=image.shape
    A=[i*8 for i in range(33)]    
    B=[j*8 for j in range(33)]
    k=1
    
    bitstream=""    
    Reconstructed_image=np.zeros(image.shape)
    for i in range(len(A)-1):
       l=1
       for j in range(len(B)-1):
          block=image[A[i]:A[k],B[j]:B[l]]
          DCT=ComputDCT(block)
          rounded_off=np.rint(DCT)
          bitstream=bitstream+encode(rounded_off.astype(int))
          Reconstructed_image[A[i]:A[k],B[j]:B[l]]=InverseDCT(rounded_off)
          l+=1
       k+=1 
    
    plt.imshow(Reconstructed_image,cmap='gray')
    plt.show()
    
    MSE=mse(image,Reconstructed_image)
    print("mean squared error is:", MSE)   
    print("Size of output bitstream is :",str(len(bitstream))+"bits")   


def problem3():
    image=skimage.io.imread(Path)
    h,w=image.shape
    A=[i*8 for i in range(33)]    #
    B=[j*8 for j in range(33)]
    
    S=list(np.linspace(10,100,10))
    
    MSE_=40                            #Just Intitalization
   
    a_=0
    b_=0
    c_=0
    for a in S:
        for b in S:
            for c in S:
                k=1
                
                bitstream=""
                
                Reconstructed_image=np.zeros(image.shape)
                Q=[[c,a,a,a,b,b,b,b],
                    [a,a,a,a,b,b,b,b],
                    [a,a,a,a,b,b,b,b],
                    [a,a,a,a,b,b,b,b],
                    [b,b,b,b,b,b,b,b],
                    [b,b,b,b,b,b,b,b],
                    [b,b,b,b,b,b,b,b],
                    [b,b,b,b,b,b,b,b]]
                for i in range(len(A)-1):
                   l=1
                   for j in range(len(B)-1):
                       block=image[A[i]:A[k],B[j]:B[l]]
                       DCT=ComputDCT(block)
                       Reconstructed_image[A[i]:A[k],B[j]:B[l]]= InverseDCT((np.multiply(Quantize(DCT,Q),Q)))
                       bitstream=bitstream+encode(Quantize(DCT,Q).astype(int))
                       l+=1
                   k+=1 
                
                MSE=mse(image,Reconstructed_image)
                print("length:",len(bitstream))
                print("mean squared error:",MSE)
                if (MSE<31.46 and len(bitstream)<=115208):
                    if(MSE<MSE_):
                        MSE_=MSE
                        a_=a
                        b_=b
                        c_=c
    
                    
    # Now using optimal a , b , c computing reconstructed image and output bitstream.
    print(a_,b_,c_)
    a=a_
    b=b_
    c=c_
    k=1
    bitstream=""
    Reconstructed_image=np.zeros(image.shape)
    Q=[[c,a,a,a,b,b,b,b],
       [a,a,a,a,b,b,b,b],
       [a,a,a,a,b,b,b,b],
       [a,a,a,a,b,b,b,b],
       [b,b,b,b,b,b,b,b],
       [b,b,b,b,b,b,b,b],
       [b,b,b,b,b,b,b,b],
       [b,b,b,b,b,b,b,b]]
    for i in range(len(A)-1):
        l=1
        for j in range(len(B)-1):
            block=image[A[i]:A[k],B[j]:B[l]]
            DCT=ComputDCT(block)
            Reconstructed_image[A[i]:A[k],B[j]:B[l]]= InverseDCT((np.multiply(Quantize(DCT,Q),Q)))
            bitstream=bitstream+encode(Quantize(DCT,Q).astype(int))
            l+=1
        k+=1         
        
        
    
    plt.imshow(Reconstructed_image,cmap='gray')
    plt.show()
    
    MSE=mse(image,Reconstructed_image)
    print("MSE:",MSE)
    print("Size of output bitstream is :",str(len(bitstream))+"bits")                                            
                
   

if __name__ == "__main__":
    problem1()
    problem2()
    problem3()
