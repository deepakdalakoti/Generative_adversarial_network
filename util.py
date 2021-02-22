import tensorflow as tf
from keras import backend as K
from keras.utils import conv_utils
from keras.layers.convolutional import UpSampling3D
from keras.engine import InputSpec
from tensorlayer.layers import *
import h5py as h5
import matplotlib.pyplot as plt
import os
import sys
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import datetime
import glob
from scipy.io import FortranFile

class DataLoader_s3d():

    def __init__(self, data_loc, nx, ny, nz, nspec, batch_size, boxsize, a_ref = 347.2):

        self.data_loc = data_loc
        self.dims = [nx, ny, nz]
        self.nspec = nspec
        self.batch_size = batch_size
        if(np.sum([ x%boxsize for x in self.dims]) > 0 ):
            sys.exit("Data dimension {}*{}*{} not divisible by boxsize {}".format(nx, ny, nz, boxsize))
        self.a_ref = a_ref
        self.boxsize = boxsize
        self._get_file_list()
        self.nbatches = int(nx*ny*nz*len(self.flist)/(boxsize**3*batch_size))
        #self.nfile_per_batch = int(boxsize**3*batch_size/(nx*ny*nz)) + 1
        self.floc = 0
        self.bloc = 0

    def _get_file_list(self):

        self.flist  = sorted(glob.glob(self.data_loc+'/field*'))

    def readfile(self, loc):

        f = FortranFile(self.flist[loc])
        nx, ny, nz = self.dims[0], self.dims[1], self.dims[2]
        time=f.read_reals(np.double)
        time_step=f.read_reals(np.double)
        time_save=f.read_reals(np.double)
        
        for L in range(self.nspec):
           tmp=f.read_reals(np.single)

        tmp=f.read_reals(np.single)
        tmp=f.read_reals(np.single)
        data = np.empty([nx, ny, nz, 3])
        for i in range(3):
            data[:,:,:,i]=f.read_reals(np.single).reshape((nx,ny,nz),order='F')*self.a_ref

        return data

    def reshape_array(self, shape, data):

        nrows = int(data.shape[0]/shape[0])
        ncols = int(data.shape[1]/shape[1])
        ndims = int(data.shape[2]/shape[2])

        reshaped = np.array([data[i*shape[0]:(i+1)*shape[0], \
                j*shape[1]:(j+1)*shape[1],k*shape[2]:(k+1)*shape[2], :] for (i,j,k) in np.ndindex(nrows, ncols, ndims)])

        return reshaped


    def getData(self, key):

        if(key > self.nbatches-1):
            raise IndexError("Index out of maximum possible batches")
        data = np.empty([self.batch_size, self.boxsize, self.boxsize, self.boxsize, 3])
        ist=0
        while True:
            tmp = self.readfile(self.floc)
            tmp = self.reshape_array([self.boxsize, self.boxsize, self.boxsize],tmp)
            nsamp = min(self.batch_size-ist, tmp.shape[0]-self.bloc)
            data[ist:nsamp+ist,:,:,:,:] = tmp[self.bloc:nsamp+self.bloc,:,:,:,:]
            ist = ist+nsamp
            self.bloc = self.bloc+nsamp
            if(self.bloc==tmp.shape[0]):
                self.bloc=0
                self.floc=self.floc+1
                if(self.floc == len(self.flist)):
                    self.floc=0

            if(ist==self.batch_size):
                break

        return data[:,:,:,:,0:1]

    def getData2(self):
        tmp = self.readfile(self.floc)
        return tmp

class UpSampling3D(Layer):
   
    def __init__(self, size=(2, 2, 2), **kwargs):
        self.size = conv_utils.normalize_tuple(size, 3, 'size')
        self.input_spec = InputSpec(ndim=5)
        super(UpSampling3D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        dim1 = self.size[0] * input_shape[1] if input_shape[1] is not None else None
        dim2 = self.size[1] * input_shape[2] if input_shape[2] is not None else None
        dim3 = self.size[2] * input_shape[3] if input_shape[3] is not None else None
        return (input_shape[0],
                dim1,
                dim2,
                dim3,
                input_shape[4])

    def call(self, inputs):
        return K.resize_volumes(inputs,
                                self.size[0], self.size[1], self.size[2],
                                self.data_format)

    def get_config(self):
        config = {'size': self.size,
                  'data_format': self.data_format}
        base_config = super(UpSampling3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
        
def subPixelConv3d(net, height_hr, width_hr, depth_hr, stepsToEnd, n_out_channel):
    """ pixle-shuffling for 3d data"""
    i = net
    r = 2
    a, b, z, c = int(height_hr/ (2 * stepsToEnd)), int(width_hr / (2 * stepsToEnd)), int(
        depth_hr / (2 * stepsToEnd)), tf.shape(i)[4]
    batchsize = tf.shape(i)[0]  # Handling Dimension(None) type for undefined batch dim
    xs = tf.split(i, r, 4)  # b*h*w*d*r*r*r
    xr = tf.concat(xs, 3)  # b*h*w*(r*d)*r*r
    xss = tf.split(xr, r, 4)  # b*h*w*(r*d)*r*r
    xrr = tf.concat(xss, 2)  # b*h*(r*w)*(r*d)*r
    x = tf.reshape(xrr, (batchsize, r * a, r * b, r * z, n_out_channel))  # b*(r*h)*(r*w)*(r*d)*n_out 

    return x



def DataLoader(datapath,filter_nr,datatype,idx,batch_size,boxsize):
    f=h5.File(datapath,'r')
    temp=0
    if datatype=='lr_train':
        lr_train = tf.Variable(tf.zeros([batch_size,boxsize,boxsize,boxsize,1] )) 
        temp=0
        count=0
        path = 'kc_000'+filter_nr+'/ps'
        for i in range (0,int(1024/boxsize)):
            for j in range (0,int(1024/boxsize)):
                for k in range (0,int(1024/boxsize)): 
                    count=count+1
                    if (int((count-1)/batch_size))==idx:
                        box = f[path][boxsize*i:boxsize*(i+1), boxsize*j:boxsize*(j+1), boxsize*k:boxsize*(k+1)] 
                        K.set_value(lr_train[temp,0:boxsize,0:boxsize,0:boxsize,0],box)
                        temp = temp+1
                        if temp==batch_size:
                            break
        return lr_train

    elif datatype=='lr_test':
        lr_test = tf.Variable(tf.zeros([int(batch_size/2),boxsize,boxsize,boxsize,1] )) 
        temp=0
        count=0
        path = 'kc_000'+filter_nr+'/ps'
        for i in range(0,int(1024/boxsize)):
            
            for j in range(0, int(1024/boxsize/2)):
                
                for k in range(60,int(1024/boxsize)):
                    count=count+1
                    if (int(2*(count-1)/batch_size))==idx:
                        box = f[path][boxsize*i:boxsize*(i+1), boxsize*j:boxsize*(j+1), boxsize*k:boxsize*(k+1)] 
                        K.set_value(lr_test[temp,0:boxsize,0:boxsize,0:boxsize,0],box)
                        temp = temp+1
                        if temp==batch_size/2:
                            break
             
        return lr_test


    elif datatype=='hr_train':
        hr_train = tf.Variable(tf.ones([batch_size,boxsize,boxsize,boxsize,1] )) 
        temp=0
        count=0
        path = '/ps/ps_01'
        for i in range (0,int(1024/boxsize)):
            for j in range (0,int(1024/boxsize)):
                for k in range (0,int(1024/boxsize)):
                    count=count+1
                    if (int((count-1)/batch_size))==idx:
                        box = f[path][boxsize*i:boxsize*(i+1), boxsize*j:boxsize*(j+1), boxsize*k:boxsize*(k+1)] 
                        K.set_value(hr_train[temp,0:boxsize,0:boxsize,0:boxsize,0],box)
                        temp = temp+1
                        if temp==batch_size:
                            break
        #print(K.eval(hr_train))
        return hr_train

    elif datatype=='hr_test':
        hr_test = tf.Variable(tf.ones([int(batch_size/2),boxsize,boxsize,boxsize,1] )) 
        temp=0
        count=0
        path = '/ps/ps_01'
        for i in range(0,int(1024/boxsize)):
            
            for j in range(0, int(1024/boxsize/2)):
                
                for k in range(60,int(1024/boxsize)):
                    count=count+1
                    if (int(2*(count-1)/batch_size))==idx:
                        box = f[path][boxsize*i:boxsize*(i+1), boxsize*j:boxsize*(j+1), boxsize*k:boxsize*(k+1)] 
                        K.set_value(hr_test[temp,0:boxsize,0:boxsize,0:boxsize,0],box)
                        temp = temp+1
                        if temp==batch_size/2:
                            break
             
        return hr_test
      

def RandomLoader_train(datapath,filter_nr,batch_size):
    boxsize=16
    f=h5.File(datapath,'r')
    idx = np.random.randint(0, 200000, batch_size) 
    start=datetime.datetime.now()
    if True:
        lr_train = tf.Variable(tf.zeros([batch_size,boxsize,boxsize,boxsize,1])) 
        hr_train = tf.Variable(tf.zeros([batch_size,boxsize,boxsize,boxsize,1])) 
        boxes=1024/boxsize
        path_lr = 'kc_000'+filter_nr+'/ps'
        path_hr = 'ps/ps_01'
        for m in range(batch_size):
            i=int(idx[m]%boxes)
            j=int((idx[m]%(boxes*boxes))/boxes)   
            k=int(idx[m]/boxes/boxes)             
            #print("i:",i,",j:",j,",k:",k)
            box_lr = f[path_lr][boxsize*i:boxsize*(i+1), boxsize*j:boxsize*(j+1), boxsize*k:boxsize*(k+1)] 
            K.set_value(lr_train[m,:,:,:,0],box_lr)
            box_hr = f[path_hr][boxsize*i:boxsize*(i+1), boxsize*j:boxsize*(j+1), boxsize*k:boxsize*(k+1)] 
            K.set_value(hr_train[m,:,:,:,0],box_hr)
   
        print("   >>training batch loaded in ",datetime.datetime.now()-start)
        return lr_train, hr_train
     
def RandomLoader_test(datapath,filter_nr,batch_size):
    boxsize=16
    f=h5.File(datapath,'r')
    idx = np.random.randint(250000, 262144, int(batch_size/2)) 
    start=datetime.datetime.now()
    if True:
        lr_test = tf.Variable(tf.zeros([int(batch_size/2),boxsize,boxsize,boxsize,1],dtype=tf.float64 )) 
        hr_test = tf.Variable(tf.zeros([int(batch_size/2),boxsize,boxsize,boxsize,1],dtype=tf.float64 )) 
        sample_lr=list()
        sample_hr=list()
        boxes=1024/boxsize
        path_lr = 'kc_000'+filter_nr+'/ps'
        path_hr = 'ps/ps_01'
        for m in range(int(batch_size/2)):

            i=int(idx[m]%boxes)
            j=int((idx[m]%(boxes*boxes))/boxes)   
            k=int(idx[m]/boxes/boxes)             
            
            box_lr = f[path_lr][boxsize*i:boxsize*(i+1), boxsize*j:boxsize*(j+1), boxsize*k:boxsize*(k+1)] 
            K.set_value(lr_test[m,:,:,:,0],box_lr)
            box_hr = f[path_hr][boxsize*i:boxsize*(i+1), boxsize*j:boxsize*(j+1), boxsize*k:boxsize*(k+1)] 
            K.set_value(hr_test[m,:,:,:,0],box_hr)
            
        print("   >>testing batch loaded in ",datetime.datetime.now()-start)
        return lr_test, hr_test
  
   
def Image_generator(box1,box2,box3,output_name):
    fig,axs = plt.subplots(1,3)
    axs[0].imshow(box1,cmap='viridis')
    axs[0].set_title('Filtered')
    axs[1].imshow(box2,cmap='viridis')
    axs[1].set_title('PIESRGAN')
    axs[2].imshow(box3,cmap='viridis')
    axs[2].set_title('Unfiltered')
    plt.savefig(output_name)
    plt.show()
    
    

      
