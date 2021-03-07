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
import time 
from scipy.fft import fftn
import multiprocessing
import ctypes
from multiprocessing import Pool
import matplotlib.pyplot as plt
#from PIESRGAN import PIESRGAN

def do_normalisation(data,which, m1, m2):

    if(which=='std'):
        data = (data-m1)/m2
        return data
    if(which=='minmax'):
        data = (data-m1)/(m2-m1)
        return data

def do_inverse_normalisation(data,which, m1, m2):

    if(which=='std'):
        data = data*m2 + m1
        return data
    if(which=='minmax'):
        data = data*(m2-m1) + m1
        return data


class DataLoader_s3d():

    def __init__(self, data_loc, nxg, nyg, nzg, nspec, batch_size, boxsize, a_ref = 347.2):

        self.data_loc = data_loc
        self.nxg, self.nyg, self.nzg = nxg, nyg, nzg
        self._get_unmorph()
        self.nspec = nspec
        self.batch_size = batch_size
        if(np.sum([ x%boxsize for x in [self.nx, self.ny, self.nz]]) > 0 ):
            sys.exit("Data dimension {}*{}*{} not divisible by boxsize {}".format(nx, ny, nz, boxsize))
            
        self.a_ref = a_ref
        self.boxsize = boxsize
        self._get_file_list()
        self.nbatches = int(self.nx*self.ny*self.nz*len(self.flist)/(boxsize**3*batch_size))
        #self.nfile_per_batch = int(boxsize**3*batch_size/(nx*ny*nz)) + 1
        self.floc = 0
        self.bloc = 0

    def _get_file_list(self):

        self.flist  = sorted(glob.glob(self.data_loc+'/field*'))

    def _get_unmorph(self):

        procs = np.loadtxt(self.data_loc+'/unmorph.in')
        self.npx, self.npy, self.npz = int(procs[0]), int(procs[1]), int(procs[2])
        self.nx, self.ny, self.nz = self.nxg//self.npx, self.nyg//self.npy, self.nzg//self.npz

    def reset_locs(self):
        self.floc=0
        self.bloc=0

    def readfile(self, loc):

        f = FortranFile(self.flist[loc])
        nx, ny, nz = self.nx, self.ny, self.nz
        time=f.read_reals(np.double)
        time_step=f.read_reals(np.double)
        time_save=f.read_reals(np.double)
        
        for L in range(self.nspec):
           tmp=f.read_reals(np.single)

        tmp=f.read_reals(np.single)
        tmp=f.read_reals(np.single)
        data = np.empty([nx, ny, nz, 3], dtype=np.float32)
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

        return data[:,:,:,:,:]

    def getRandomData(self):
        data = np.empty([self.batch_size, self.boxsize, self.boxsize, self.boxsize, 3])
        ist=0
        loc  = np.random.randint(0,len(self.flist))
        bloc = np.random.randint(0, int(self.nx*self.ny*self.nz/self.boxsize**3))
        while True:
            tmp = self.readfile(loc)
            tmp = self.reshape_array([self.boxsize, self.boxsize, self.boxsize],tmp)
            nsamp = min(self.batch_size-ist, tmp.shape[0]-bloc)
            data[ist:nsamp+ist,:,:,:,:] = tmp[bloc:nsamp+bloc,:,:,:,:]
            ist = ist+nsamp
            bloc = bloc+nsamp
            if(bloc==tmp.shape[0]):
                bloc=0     # Although random, I want to be continous in the physical space
                loc=loc+1  # so increment by one rather than taking random nums again
                if(self.floc == len(self.flist)):
                    self.floc=0

            if(ist==self.batch_size):
                break

        return data

    def get_data_plane(self, plane, channels):
        # Only for X-Y planes
        zid = int(plane//(self.nzg/self.npz ))
        zloc = int(plane%(self.nzg/self.npz ))
        print(zid, zloc, plane)
        data = np.empty([self.nxg, self.nyg, channels])
        for yid in range(self.npy):
            for xid in range(self.npx):
                myid = zid*self.npx*self.npy + yid*self.npx + xid
                filename = self.data_loc + '/field.' + "{0:0=6d}".format(myid)
                idx = self.flist.index(filename)
                xst = xid*self.nx
                xen = (xid+1)*self.nx
                yst = yid*self.ny
                yen = (yid+1)*self.ny
                data[xst:xen,yst:yen,:]  = self.readfile(idx)[:,:,zloc,0:channels]
        return data

    def get_data_all(self):
        u = np.empty([self.nxg, self.nyg, self.nzg,3], dtype = np.float32)

        for zid in range(self.npz):
            for yid in range(self.npy):
                for xid in range(self.npx):
                    myid = zid*self.npx*self.npy + yid*self.npx + xid
                    fname = self.data_loc + '/field.' + "{0:0=6d}".format(myid)
                    xsrt=xid*self.nx
                    xend=(xid+1)*self.nx
                    ysrt=yid*self.ny
                    yend=(yid+1)*self.ny
                    zsrt=zid*self.nz
                    zend=(zid+1)*self.nz
                    idx = self.flist.index(fname)
                    t1 = time.time()
                    u[xsrt:xend, ysrt:yend, zsrt:zend,:] = self.readfile(idx)
                    print("Read " + '/field.'+ "{0:0=6d}".format(myid) + " in {} secs".format(time.time()-t1) )
        return u

    def read_parallel(self, workers):
        #ush = np.zeros([1536, 1536, 1536, 3], dtype=np.float32)
        
        shared = multiprocessing.Array('f', self.nxg*self.nyg*self.nzg*3)
        ush = np.frombuffer(shared.get_obj(), dtype=np.float32)
        ush = ush.reshape(self.nxg,self.nyg,self.nzg,3)
        print(ush.shape)
        t1=time.time()
        p = Pool(workers, initializer=self.init_shared_s3d, initargs=(ush,))
        for zid in range(self.npz):
            for yid in range(self.npy):
                for xid in range(self.npx):
                    xst, xen = xid*self.nx, (xid+1)*self.nx
                    yst, yen = yid*self.ny, (yid+1)*self.ny
                    zst, zen = zid*self.nz, (zid+1)*self.nz
                    myid = zid*self.npx*self.npy + yid*self.npx + xid
                    fname = self.data_loc + '/field.' + "{0:0=6d}".format(myid)
                    idx = self.flist.index(fname)
                    p.apply_async(self.read_single, args = (idx, xst, \
                            xen, yst, yen, zst, zen, ), error_callback= print_error)
        p.close()
        p.join()
        #plt.imshow(ush[0:500,0:500,0,0])
        #plt.savefig('testfig.png')
        #print("HERE", ush[0,0,0,0], ush[1,100,0,0])
        print("Total read time {} secs".format(time.time()-t1))
        return ush

    def init_shared_s3d(self,shared_arr_):
        global ush
        ush = shared_arr_

    def read_single(self,idx, xst, xen, yst, yen, zst, zen):
        t1 = time.time()
        global ush
        ush[xst:xen,yst:yen, zst:zen,:] = self.readfile(idx)
        #ush[0,0,0,0] = idx
        #print(ush[0,0,0,0])
        print("Read file in {} secs".format(time.time()-t1))
        return


    def get_norm_constants(self):

        if(os.path.isfile('./mins.dat')):
            mins = np.loadtxt('mins.dat')
            maxs = np.loadtxt('maxs.dat')
            self.mins = mins
            self.maxs = maxs
            return mins, maxs

        means = np.zeros(3)
        stds = np.zeros(3)
        mins = np.zeros(3)
        maxs = np.zeros(3)

        for i in range(self.nbatches):
            data = self.readfile(i)
            mins = np.minimum(mins, np.min(data,axis=(0,1,2)))
            maxs = np.maximum(maxs, np.max(data,axis=(0,1,2)))
        np.savetxt('./mins.dat', mins)
        np.savetxt('./maxs.dat', maxs)
        self.mins = mins
        self.maxs = maxs
        return mins, maxs

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
    fig,axs = plt.subplots(1,3, figsize=(15,15))
    cmap = 'seismic'
    #axs[0].contourf(box1, levels=40)
    '''
    plt.subplot(1,3,1)
    plt.title('Filtered')
    plt.contourf(box1)
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.title('PIESRGAN')
    plt.contourf(box2)
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.title('DNS')
    plt.contourf(box3)
    plt.colorbar()
    plt.tight_layout()
    '''
    im=axs[1].imshow(box2,cmap=cmap)
    clim = im.properties()['clim']
    axs[0].imshow(box1,cmap=cmap, clim=clim)
    axs[0].set_title('Filtered')
    #axs[1].contourf(box2, levels=40)
    axs[1].set_title('PIESRGAN')
    axs[2].imshow(box3,cmap=cmap, clim=clim)
    #axs[2].contourf(box3, levels=40)
    axs[2].set_title('Unfiltered')
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)
    plt.savefig(output_name, dpi=500)
    plt.show()
    
def read_single(readfile, idx, xst, xen, yst, yen, zst, zen):
        t1 = time.time()
        ush[xst:xen,yst:yen, zst:zen,:] = readfile(idx)
        #ush[0,0,0,0] = idx
        #print(ush[0,0,0,0])
        print("Read file in {} secs".format(time.time()-t1))
        return

def print_error(err):
    print(err)
    return
