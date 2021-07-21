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
import scipy
import multiprocessing
import ctypes
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pickle

''' Utilities to support training GAN using s3d data '''


def do_normalisation(data,which, m1, m2):
    # Normalise data 
    if(which=='std'):
        data = (data-m1)/m2
        return data
    if(which=='minmax'):
        data = (data-m1)/(m2-m1)
        return data
    if(which=='range'):
        data = data/(m2-m1)
        return data

def do_inverse_normalisation(data,which, m1, m2):
    # Inverse normalisation
    if(which=='std'):
        data = data*m2 + m1
        return data
    if(which=='minmax'):
        data = data*(m2-m1) + m1
        return data
    if(which=='range'):
        data = data*(m2-m1)
        return data


class DataLoader_s3d():

    def __init__(self, data_loc, nxg, nyg, nzg, nspec, batch_size, boxsize, channels, a_ref = 347.2):
        # A class to read and manipulate s3d savefile data
        # Arguments are self explanatory for s3d dataset
        # boxsize and batch_size relate to the size of dataset for CNN training
        # channels =1 will give one component of velocity, 2 will give 2 and so on
        self.data_loc = data_loc
        self.nxg, self.nyg, self.nzg = nxg, nyg, nzg
        self._get_unmorph()
        self.nspec = nspec
        self.batch_size = batch_size
        self.channels = channels
        #if(np.sum([ x%boxsize for x in [self.nx, self.ny, self.nz]]) > 0 ):
        #    sys.exit("Data dimension {}*{}*{} not divisible by boxsize {}".format(self.nx, self.ny, self.nz, boxsize))
            
        self.a_ref = a_ref
        self.boxsize = boxsize
        self._get_file_list()
        self.nbatches = int(self.nx*self.ny*self.nz*len(self.flist)/(boxsize**3*batch_size))
        self.nbatches_plane = int(self.nxg*self.nyg*self.boxsize/(boxsize**3*batch_size))
        #self.nbatches_plane = int(256*256*self.boxsize/(boxsize**3*batch_size))
        #self.nfile_per_batch = int(boxsize**3*batch_size/(nx*ny*nz)) + 1
        self.floc = 0
        self.bloc = 0
        self.init_train=0

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
        # Read and store velocity data, for one file in s3d savfile
        f = FortranFile(self.flist[loc])
        nx, ny, nz = self.nx, self.ny, self.nz
        time=f.read_reals(np.double)
        time_step=f.read_reals(np.double)
        time_save=f.read_reals(np.double)
        
        for L in range(self.nspec):
           tmp=f.read_reals(np.single)

        tmp=f.read_reals(np.single)
        tmp=f.read_reals(np.single)
        data = np.empty([nx, ny, nz, self.channels], dtype=np.float32)
        for i in range(self.channels):
            data[:,:,:,i]=f.read_reals(np.single).reshape((nx,ny,nz),order='F')*self.a_ref

        return data

    def reshape_array(self, shape, data):

        nrows = int(data.shape[0]/shape[0])
        ncols = int(data.shape[1]/shape[1])
        ndims = int(data.shape[2]/shape[2])

        reshaped = np.array([data[i*shape[0]:(i+1)*shape[0], \
                j*shape[1]:(j+1)*shape[1],k*shape[2]:(k+1)*shape[2], :] for (i,j,k) in np.ndindex(nrows, ncols, ndims)])

        return reshaped

    def inverse_reshape_array(self, data):
        batches = data.shape[0]
        data_res = np.empty([data.shape[1], data.shape[2]*batches, data.shape[3],data.shape[4]],dtype=np.float32)
        for i in range(batches):
            data_res[:,i*data.shape[1]:(i+1)*data.shape[1],:,:] = data[i,:,:,:,:]
        return data_res

    def getData(self, key):
        # Read a data corresponding to the input boxsize and batch_size defined in the class
        # key value will range between 0 and nbatches-1 inclusive
        ''' Good thing is that this remebers where we left previously and will always give non overlapping data'''
        if(key > self.nbatches-1):
            raise IndexError("Index out of maximum possible batches")
        data = np.empty([self.batch_size, self.boxsize, self.boxsize, self.boxsize, self.channels], dtype=np.float32)
        nbatches_per_file = int(self.nx*self.ny*self.nz/(self.boxsize**3))
        nbatches_total = key*self.batch_size
        floc =  int(nbatches_total/nbatches_per_file) # Which file we are at
        nst  =  key*self.batch_size - floc*nbatches_per_file # Where in that file to start
        ist=0
        while True:
            tmp = self.readfile(floc)
            tmp = self.reshape_array([self.boxsize, self.boxsize, self.boxsize],tmp)
            nsamp = min(self.batch_size-ist, tmp.shape[0]-nst)
            data[ist:nsamp+ist,:,:,:,:] = tmp[nst:nsamp+nst,:,:,:,:]
            ist = ist+nsamp
            nst = nst+nsamp
            if(nst==tmp.shape[0]):
                nst=0
                floc=floc+1
                if(floc == len(self.flist)):
                   floc=0

            if(ist==self.batch_size):
                break

        return data

    def getTrainData(self, idx):
        # Read a data corresponding to the input boxsize and batch_size defined in the class
        ''' The difference between this and getData is that this one will, when first called will read all
            data in memory and then for subsequent calls will return subarrays of the data. This is probably not the
            best way to handle this but the reasoning for doing this is as follows
            
            In the current case we upsample data for a particular upscaling factor
            Now consider that the chosen boxsize for LES data is 16, for upscaling factor of 8
            the DNS boxsize will be 96. The function should then return boxes of size [96, 96, 96].
            Based on the processor topology for the current Isotropic turbulence data, one datafile field.xxxx
            has the size [96, 96, 64], so for getting a data of size [96, 96, 96] we will have to read multiple files 
            and then remember what files have been read and upto what point so that when next time the data is requested
            it is not repeated. This can of course be done but I didnt do it yet.'''
        if(idx > self.nbatches-1):
            raise IndexError("Index out of maximum possible batches")
        
        if(not self.init_train):
            print(self.init_train,"INIT")
            udata  = self.read_parallel(48)
            self.udata = self.reshape_array([self.boxsize, self.boxsize, self.boxsize], udata)

            self.init_train=1

        return self.udata[idx*self.batch_size:(idx+1)*self.batch_size,:,:,:,:]

    def getTrainData_plane(self, workers, plane, idx):
        # Same as above but the Z planes don't change and Z index has size of boxsize
        if(idx > self.nbatches_plane-1):
            raise IndexError("Index out of maximum possible batches")
        
        if(not self.init_train):
            print(self.init_train,"INIT")
            self.udata = np.zeros([self.nxg,self.nyg,self.boxsize,self.channels], dtype=np.float32)
            slo = int(max(plane-self.boxsize/2,0))
            shi = int(min(self.boxsize+slo,self.nzg-1))
            if(shi-slo < self.boxsize):
                slo = int(self.boxsize-(shi-slo))

            res=[]

            for i in range(slo,shi):
                self.udata[:,:,i-slo,:] = self.get_data_plane(i)
            self.udata = self.reshape_array([self.boxsize, self.boxsize, self.boxsize], self.udata)
            self.init_train=1

        return self.udata[idx*self.batch_size:(idx+1)*self.batch_size,:,:,:,:]

    def getRandomData(self):
        # Get random data in correct shape and topologically correct way
        data = np.empty([self.batch_size, self.boxsize, self.boxsize, self.boxsize, self.channels])
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

    def get_data_plane(self, plane):
        # Only for X-Y planes

        zid = int(plane//(self.nzg/self.npz ))
        zloc = int(plane%(self.nzg/self.npz ))
        print("Reading plane {}".format(plane))
        data = np.empty([self.nxg, self.nyg, self.channels])
        for yid in range(self.npy):
            for xid in range(self.npx):
                myid = zid*self.npx*self.npy + yid*self.npx + xid
                filename = self.data_loc + '/field.' + "{0:0=6d}".format(myid)
                idx = self.flist.index(filename)
                xst = xid*self.nx
                xen = (xid+1)*self.nx
                yst = yid*self.ny
                yen = (yid+1)*self.ny
                data[xst:xen,yst:yen,:]  = self.readfile(idx)[:,:,zloc,0:self.channels]
        return data


    def read_parallel(self, workers):
        # Use muliprocessing to read the s3d savefile 
        shared = multiprocessing.Array('f', self.nxg*self.nyg*self.nzg*self.channels)
        ush = np.frombuffer(shared.get_obj(), dtype=np.float32)
        ush = ush.reshape(self.nxg,self.nyg,self.nzg,self.channels)
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
        print("Total read time {} secs".format(time.time()-t1))
        return ush

    def init_shared_s3d(self,shared_arr_):
        global ush
        ush = shared_arr_

    def read_single(self,idx, xst, xen, yst, yen, zst, zen):
        t1 = time.time()
        global ush
        ush[xst:xen,yst:yen, zst:zen,:] = self.readfile(idx)
        print("Read file in {} secs".format(time.time()-t1))
        return


    def get_norm_constants(self, workers):
        # Get and save normalisation constants if not already obtained
        if(os.path.isfile('./norm_const.dat')):
            const = np.loadtxt('norm_const.dat')
            self.mins = const[0]
            self.maxs = const[1]
            self.mean = const[2]
            self.std  = const[3]
            return 

        const = np.zeros(4)

        # For isotropic turbulence, I am taking same mean for all data
        data = self.read_parallel(workers)
        const[0] = np.min(data)
        const[1] = np.max(data)
        const[2] = np.mean(data)
        const[3] = np.sqrt(np.mean(data**2)-const[2]**2)

        np.savetxt('./norm_const.dat',const)
        self.mins = const[0]
        self.maxs = const[1]
        self.mean = const[2]
        self.std  = const[3]
        return 

    def filter_data(self, fact, data, filt='box'):
        # Filter data using a gaussian/box filter
        dimx = int(data.shape[0]/fact)
        dimy = int(data.shape[1]/fact)
        dimz = int(data.shape[2]/fact)

        wght = np.zeros([fact, fact, fact], dtype=np.float32)
        if(filt=='gaussian'):
            ilo = -fact/2 + 0.5
            sigma = fact**2
            for i in range(fact):
                for j in range(fact):
                    for k in range(fact):
                        idx, idy, idz = ilo + i, ilo+j, ilo+k
                        wght[i,j,k] = np.exp(-(idx**2 + idy**2 + idz**2)/sigma)
        else:
            wght[:,:,:] = 1.0

        wght = wght/np.sum(wght)

        data_out = np.zeros([dimx, dimy, dimz, data.shape[3]], dtype=np.float32)
        for i in range(dimx):
            for j in range(dimy):
                for k in range(dimz):
                    ilo = i*fact
                    ihi = (i+1)*fact
                    jlo = j*fact
                    jhi = (j+1)*fact
                    klo, khi = k*fact, (k+1)*fact
                    #data_out[i,j,k,:] = np.mean(data[ilo:ihi, jlo:jhi,klo:khi,:])
                    data_out[i,j,k,:] = np.sum(np.multiply(data[ilo:ihi, jlo:jhi, klo:khi, :],wght[:,:,:,None]))

        return data_out

    def filter_data_2d(self, fact, data, sigma, filt='box'):
        # Same as above but for 2D data
        dimx = int(data.shape[0]/fact)
        dimy = int(data.shape[1]/fact)

        wght = np.zeros([fact, fact], dtype=np.float32)
        if(filt=='gaussian'):
            ilo = -fact/2 + 0.5
            for i in range(fact):
                for j in range(fact):
                        idx, idy = ilo + i, ilo+j
                        wght[i,j] = np.exp(-(idx**2 + idy**2)/sigma)
        else:
            wght[:,:] = 1.0

        wght = wght/np.sum(wght)
        data_out = np.zeros([dimx, dimy, data.shape[2]])

        for i in range(dimx):
            for j in range(dimy):
                    ilo = i*fact
                    ihi = (i+1)*fact
                    jlo = j*fact
                    jhi = (j+1)*fact
                    #data_out[i,j,:] = np.mean(data[ilo:ihi, jlo:jhi,:])
                    data_out[i,j,:] = np.sum(np.multiply(data[ilo:ihi, jlo:jhi, :], wght[:,:,None]))
        return data_out

    def interpolate_2d(self, fact, data):
        # linearly interpolate from grid described by data to grid which is smaller by a factor of fact in each dir
        dimx = int(data.shape[0]/fact)
        dimy = int(data.shape[1]/fact)
        x = np.linspace(0,5e-3, self.nxg)
        y = np.linspace(0,5e-3, self.nyg)

        xi = np.linspace(0, 5e-3, dimx)
        yi = np.linspace(0, 5e-3, dimy)
        X, Y = np.meshgrid(xi, yi, indexing ='ij')
        X = np.reshape(X, [dimx*dimx, -1])
        Y = np.reshape(Y, [dimx*dimx, -1])

        
        data_out = scipy.interpolate.interpn((x,y), data, (X,Y))
        data_out =  np.reshape(data_out, [dimx, -1])
        print(data_out.shape)

        return data_out

    def smooth_2d(self, fact, data):
        # Smooth 2D data by smoothing 
        data_out = np.zeros_like(data)
        # Check
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ilo = max(0, int(i-fact/2))
                ihi = min(data.shape[0], int(i+fact/2))

                jlo = max(0, int(j-fact/2))
                jhi = min(data.shape[0], int(j+fact/2))


                data_out[i,j,:] = np.mean(data[ilo:ihi, jlo:jhi,:], axis=(0,1))

        return data_out

    def smooth_3d(self, fact, data):
        # Same as above
        data_out = np.zeros_like(data)
        # Check
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    ilo = max(0, int(i-fact/2))
                    ihi = min(data.shape[0], int(i+fact/2))

                    jlo = max(0, int(j-fact/2))
                    jhi = min(data.shape[1], int(j+fact/2))

                    klo = max(0, int(k-fact/2))
                    khi = min(data.shape[2], int(k+fact/2))


                    data_out[i,j,k,:] = np.mean(data[ilo:ihi, jlo:jhi,klo:khi,:], axis=(0,1,2))

        return data_out

class write_all_wghts(tf.keras.callbacks.Callback):
    ''' Write model weights and optimizer state
        Custom callback, better than default tensorflow save weights because also saves
        optimizer state'''
       
    def __init__(self,write_freq, write_dir, prefix, epoch):
        self.write_freq=write_freq
        self.write_dir = write_dir
        self.prefix = prefix
        self.epoch = epoch
        try:
            os.makedirs(self.write_dir)
            print("Created {} directory".format(self.write_dir))
        except FileExistsError:
            print("Directory {} already exists".format(self.write_dir))
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch = self.epoch+1
        if(self.epoch%self.write_freq==0):
            self.model.save_weights(self.write_dir+self.prefix+'_{}.h5'.format(self.epoch))
            opt_wght = self.model.optimizer.get_weights()
            pickle.dump(opt_wght,open(self.write_dir+self.prefix+'_opt_idx_{}.h5'.format(self.epoch),'wb'))

class tensorboard_stats(tf.keras.callbacks.Callback):
    ''' Write stats for tensorboard '''
    def __init__(self, write_freq, write_dir, epoch):
        self.write_freq=write_freq
        self.write_dir = write_dir
        self.epoch = epoch
        try:
            os.makedirs(self.write_dir)
            print("Created {} directory".format(self.write_dir))
        except FileExistsError:
            print("Directory {} already exists".format(self.write_dir))
        
        self.writer = tf.summary.create_file_writer(self.write_dir)
    def on_epoch_end(self, epoch, logs=None):
        self.epoch = self.epoch+1
        if(self.epoch%self.write_freq==0):
            with self.writer.as_default():
                for idx in logs:
                    tf.summary.scalar(idx, logs[idx], step= self.epoch)
                for weights in self.model.trainable_weights:
                    tf.summary.histogram(weights.name, data=weights, step=self.epoch)

def subPixelConv3d(net, height_hr, width_hr, depth_hr, stepsToEnd, n_out_channel):
    """ pixle-shuffling for 3d data
        This upsamples data by a factor of 2 in each dir"""

    i = net
    r = 2
    a, b, z, c = int(height_hr/ (2 ** stepsToEnd)), int(width_hr / (2 ** stepsToEnd)), int(
        depth_hr / (2 ** stepsToEnd)), tf.shape(i)[4]
    batchsize = tf.shape(i)[0]  # Handling Dimension(None) type for undefined batch dim
    xs = tf.split(i, r, 4)  # b*h*w*d*r*r*r
    xr = tf.concat(xs, 1)  # b*h*w*(r*d)*r*r
    xss = tf.split(xr, r, 4)  # b*h*w*(r*d)*r*r
    xrr = tf.concat(xss, 2)  # b*h*(r*w)*(r*d)*r
    xsss = tf.split(xrr, r, 4)
    xrrr = tf.concat(xsss,3)
    x = tf.reshape(xrrr, (batchsize, r * a, r * b, r * z, n_out_channel))  # b*(r*h)*(r*w)*(r*d)*n_out 

    return x

def phaseShift(inputs, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 4, 2, 5, 3, 6])

    return tf.reshape(X, shape_2)


# The implementation of PixelShuffler
def pixelShuffler(inputs, scale=2):
    ''' Another way to upsample data '''
    size = inputs.get_shape().as_list()
    #batch_size = size[0]
    batch_size = tf.shape(inputs)[0]
    d = size[1]
    h = size[2]
    w = size[3]
    c = size[4]

    # Get the target channel size
    channel_target = c // (scale * scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, d, h, w, scale, scale, scale]
    shape_2 = [batch_size, d * scale, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=4)
    output = tf.concat([phaseShift(x, shape_1, shape_2) for x in input_split], axis=4)

    return output


def Image_generator(box1,box2,box3,output_name):
    # Handly function to plot data
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
    axs[0].set_title('LES')
    #axs[1].contourf(box2, levels=40)
    axs[1].set_title('GAN')
    axs[2].imshow(box3,cmap=cmap, clim=clim)
    #axs[2].contourf(box3, levels=40)
    axs[2].set_title('DNS')
    axs[0].set_yticklabels([])
    axs[1].set_yticklabels([])
    axs[2].set_yticklabels([])


    axs[0].set_xticklabels([])
    axs[1].set_xticklabels([])
    axs[2].set_xticklabels([])

    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.25)
    plt.savefig(output_name, dpi=500)
    plt.close()

def Image_generator2(box1,box2,box3,output_name):
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
    axs[0].set_title('LES')
    #axs[1].contourf(box2, levels=40)
    axs[1].set_title('GAN')
    axs[2].imshow(box3,cmap=cmap, clim=clim)
    #axs[2].contourf(box3, levels=40)
    axs[2].set_title('DNS')

    #axs[0].set_yticklabels([])
    axs[1].set_yticklabels([])
    axs[2].set_yticklabels([])


    #axs[0].set_xticklabels([])
    axs[1].set_xticklabels([])
    axs[2].set_xticklabels([])


    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)
    plt.savefig(output_name, dpi=500)
    plt.close()

def Image2(box1,box2,output_name):
    fig,axs = plt.subplots(1,2, figsize=(15,15))
    cmap = 'seismic'
    im=axs[1].imshow(box2,cmap=cmap)
    clim = im.properties()['clim']
    axs[0].imshow(box1,cmap=cmap, clim=clim)
    axs[0].set_title('LES')
    #axs[1].contourf(box2, levels=40)
    axs[1].set_title('GAN')

    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.25)
    plt.savefig(output_name, dpi=500)
    plt.close()

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


