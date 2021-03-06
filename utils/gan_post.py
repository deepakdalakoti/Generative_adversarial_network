from util import DataLoader_s3d, do_normalisation
from PIESRGAN import PIESRGAN
import multiprocessing 
from multiprocessing import Pool
#import ray
#from ray.util.multiprocessing import Pool
import numpy as np
import time
import dill
import os
from util import do_normalisation, do_inverse_normalisation
import pickle
from scipy.fft import fftn
from spectrum import do_spectrum, do_spectrum_slice
#import pathos.multiprocessing as pm
# TO DO : add channels to dataloader
RRDB_layers = 6

class GAN_post(DataLoader_s3d):

    def __init__(self,data_loc, nxg, nyg, nzg, nspec, batch_size, boxsize, channels, weights, a_ref = 347.2):

        self.weights=weights
        super().__init__(data_loc, nxg, nyg, nzg, nspec, batch_size, boxsize, channels, a_ref)
        self.get_norm_constants(48)

    def get_pred_parallel(self, workers):
        shared = multiprocessing.Array('f', self.nxg*self.nyg*self.nzg*self.channels)
        upred = np.frombuffer(shared.get_obj(), dtype=np.float32)
        upred = upred.reshape(self.nxg,self.nyg,self.nzg,self.channels)
        t1=time.time()
        res=[]
        p = Pool(workers, initializer = self.init_shared, initargs=(upred,))
        for fnum in range(len(self.flist)):
            data = self.readfile(fnum)
            data = self.reshape_array([self.boxsize, self.boxsize, self.boxsize],data)


            r=p.apply_async(self.pred_gan,args=(self.boxsize,data,),  error_callback=self.print_error)
            res.append([r,fnum])

        p.close()
        p.join()

        for i in range(len(self.flist)):
            fnum = res[i][1]
            zid = fnum//(self.npx*self.npy)
            yid = (fnum%(self.npx*self.npy))//self.npx
            xid = (fnum%(self.npx*self.npy))%self.npx

            nbx = self.nx/self.boxsize
            nby = self.ny/self.boxsize
            nbz = self.nz/self.boxsize
            bx  = self.boxsize
            pred = res[i][0].get()
            for i in range(data.shape[0]):
                ii = int(i//(nby*nbz))
                jj = int((i%(nby*nbz))//(nbz))
                kk = int((i%(nby*nbz))%(nbz))
                xst = xid*self.nx
                yst = yid*self.ny
                zst = zid*self.nz
                upred[xst+ii*bx:xst+(ii+1)*bx, yst+jj*bx:yst+(jj+1)*bx,zst+kk*bx:zst+(kk+1)*bx,:] = pred[i,:,:,:,:]


        print("Total pred time {} secs".format(time.time()-t1))
        return upred

    def get_pred_serial(self, norm, save=None):
        upred = np.zeros([self.nxg, self.nyg, self.nzg, self.channels], dtype=np.float32)
        t1=time.time()
        res=[]

        nbx = self.nx/self.boxsize
        nby = self.ny/self.boxsize
        nbz = self.nz/self.boxsize
        bx  = self.boxsize
        gan = PIESRGAN(training_mode=False, height_lr = self.boxsize, \
                    width_lr=self.boxsize, depth_lr = self.boxsize, channels=self.channels, RRDB_layers=RRDB_layers)
        gan.load_weights(generator_weights=self.weights)

        if(norm=='minmax'):
            m1=self.mins
            m2=self.maxs
        if(norm=='std'):
            m1=self.mean
            m2=self.std
        if(norm=='range'):
            m1=self.mins
            m2=self.maxs
        

        for fnum in range(len(self.flist)):
            data = self.readfile(fnum)
            data = self.reshape_array([self.boxsize, self.boxsize, self.boxsize],data)
            t1=time.time()
            data = do_normalisation(data,norm, m1, m2)
            pred = gan.generator.predict(data, batch_size=self.batch_size)
            pred = do_inverse_normalisation(pred, norm, m1, m2)
            print("Predicted in {} secs fnum {}".format(time.time()-t1, fnum))

            zid = fnum//(self.npx*self.npy)
            yid = (fnum%(self.npx*self.npy))//self.npx
            xid = (fnum%(self.npx*self.npy))%self.npx

            for i in range(data.shape[0]):
                ii = int(i//(nby*nbz))
                jj = int((i%(nby*nbz))//(nbz))
                kk = int((i%(nby*nbz))%(nbz))
                xst = xid*self.nx
                yst = yid*self.ny
                zst = zid*self.nz
                upred[xst+ii*bx:xst+(ii+1)*bx, yst+jj*bx:yst+(jj+1)*bx,zst+kk*bx:zst+(kk+1)*bx,:] = pred[i,:,:,:,:]


        print("Total pred time {} secs".format(time.time()-t1))
        if(save is not None):
            np.save(save,upred)    
        return upred

    def get_spectrum_slice(self, workers, xmax, xmin, ymax, ymin, plane, norm, savePred=None, pref=''):
        # Get spectrum of the GAN predictions
        t1=time.time()
        upred = self.get_gan_slice_serial(plane, workers, norm)
        print("Predicted in {} secs".format(time.time()-t1))
        nt = self.nxg*self.nyg
        u_k = np.empty([self.nxg, self.nyg, self.channels], dtype=complex)
        for L in range(self.channels):
            t1 = time.time()
            u_k[:,:,L] = fftn(upred[:,:,int(self.boxsize/2),L], workers=-1)/nt
            print("FFT finished in {} secs".format(time.time()-t1))
        nhx, nhy = self.nxg/2+1, self.nyg/2+1
        nk = max([nhx, nhy])
        factx = 2.0*np.pi/(xmax-xmin)
        facty = 2.0*np.pi/(ymax-ymin)

        kx_max = (nhx-1)*factx
        ky_max = (nhy-1)*facty

        kmax = np.sqrt(kx_max**2+ky_max**2)
        del_k = kmax/(nk-1)
        wavenumbers = np.arange(0,nk)*del_k
        energy = np.zeros(int(nk))
        res = []
        p = Pool(workers)
        for xid in range(self.npx):
            for yid in range(self.npy):
                    xst, xen = xid*self.nx, (xid+1)*self.nx
                    yst, yen = yid*self.ny, (yid+1)*self.ny

                    r=p.apply_async(do_spectrum_slice, args = (self, u_k[xst:xen,yst:yen,:], \
                            xmax, xmin, ymax, ymin, xid, yid, ), \
                            error_callback= self.print_error)
                    res.append(r)

        p.close()
        p.join()
        for i in range(len(res)):
            energy = energy + res[i].get()
        spectrum = np.zeros([int(nk),2])
        spectrum[:,0] = wavenumbers
        spectrum[:,1] = energy
        np.savetxt(pref+'spectrum',spectrum)
        return spectrum


    def get_spectrum(self, workers, xmax, xmin, ymax, ymin, zmax, zmin, norm, savePred=None, pref=''):
        # Get spectrum of the GAN predictions
        t1=time.time()
        upred = self.get_pred_serial(norm,savePred)
        #upred = np.load('pred_s-2.4500E-05.pkl.npy')
        print("Predicted in {} secs".format(time.time()-t1))
        nt = self.nxg*self.nyg*self.nzg
        u_k = np.empty([self.nxg, self.nyg, self.nzg,self.channels], dtype=complex)
        for L in range(self.channels):
            t1 = time.time()
            u_k[:,:,:,L] = fftn(upred[:,:,:,L], workers=-1)/nt
            print("FFT finished in {} secs".format(time.time()-t1))
        nhx, nhy, nhz = self.nxg/2+1, self.nyg/2+1, self.nzg/2+1
        nk = max([nhx, nhy, nhz])
        factx = 2.0*np.pi/(xmax-xmin)
        facty = 2.0*np.pi/(ymax-ymin)
        factz = 2.0*np.pi/(zmax-zmin)

        kx_max = (nhx-1)*factx
        ky_max = (nhy-1)*facty
        kz_max = (nhz-1)*factz

        kmax = np.sqrt(kx_max**2+ky_max**2+kz_max**2)
        del_k = kmax/(nk-1)
        wavenumbers = np.arange(0,nk)*del_k
        energy = np.zeros(int(nk))
        res = []
        p = Pool(workers)
        for xid in range(self.npx):
            for yid in range(self.npy):
                for zid in range(self.npz):
                    xst, xen = xid*self.nx, (xid+1)*self.nx
                    yst, yen = yid*self.ny, (yid+1)*self.ny
                    zst, zen = zid*self.nz, (zid+1)*self.nz

                    r=p.apply_async(do_spectrum, args = (self, u_k[xst:xen,yst:yen,zst:zen,:], \
                            xmax, xmin, ymax, ymin, zmax, zmin,xid, yid, zid,), \
                            error_callback= self.print_error)
                    res.append(r)

        p.close()
        p.join()
        for i in range(len(res)):
            energy = energy + res[i].get()
        spectrum = np.zeros([int(nk),2])
        spectrum[:,0] = wavenumbers
        spectrum[:,1] = energy
        np.savetxt(pref+'spectrum',spectrum)
        return spectrum

    def get_gan_slice(self, plane, workers):
        self.get_norm_constants(48)
        zid  = int(plane//(self.nzg/self.npz))
        zloc = int(plane%(self.nzg/self.npz))
        #shared = multiprocessing.Array('f', self.nxg*self.nyg*16*3)
        #upred = np.frombuffer(shared.get_obj(), dtype=np.float32)
        #upred = upred.reshape(self.nxg,self.nyg,16,3)
        udata = np.zeros([self.nxg,self.nyg,self.boxsize,self.channels], dtype=np.float32)
        #ray.init(address='auto', dashboard_host='0.0.0.0', dashboard_port=8888) 
        #p = Pool(workers, initializer = self.init_shared, initargs=(upred,))
        p = Pool(workers)
        slo = int(max(plane-self.boxsize/2,0))
        shi = int(min(self.boxsize+slo,self.nzg-1))
        if(shi-slo < self.boxsize):
            slo = int(self.boxsize-(shi-slo))

        res=[]

        for i in range(slo,shi):
            r=p.apply_async(self.get_data_plane, args=(i,), error_callback=self.print_error)
            res.append([r,int(i-slo)])
        for i in range(len(res)):
            udata[:,:,res[i][1],:] = res[i][0].get()
        p.close()
        p.join()

        udata = self.reshape_array([self.boxsize, self.boxsize, self.boxsize], udata)
        
        nbatches = udata.shape[0]//self.batch_size
        res=[]
        t1=time.time()
        p = Pool(workers)
        for i in range(nbatches):
            ist = i*self.batch_size
            ien = min((i+1)*self.batch_size, udata.shape[0])
            r = p.apply_async(self.pred_gan, args=(self.boxsize, udata[ist:ien,:,:,:,:],), \
                error_callback=self.print_error)
            res.append([r,ist, ien])

        p.close()
        p.join()
        
        bx  = self.boxsize
        nby = self.nyg/bx
        upred = np.zeros([self.nxg, self.nyg, self.boxsize, self.channels], dtype=np.float32)        
        for i in range(len(res)):
            pred = res[i][0].get()
            ist  = res[i][1]
            ien  = res[i][2]
            for j in range(ist, ien):
                ii = int(j//nby)
                jj = int(j%nby)
                kk=0
                upred[ii*bx:(ii+1)*bx, jj*bx:(jj+1)*bx,kk*bx:(kk+1)*bx,:] = pred[j-ist,:,:,:,:]
        '''
        gan = PIESRGAN(training_mode=False, height_lr = self.boxsize, \
                    width_lr=self.boxsize, depth_lr = self.boxsize, channels=3)
        gan.load_weights(generator_weights='./data/weights/_generator_idx5120.h5')

        t1=time.time()
        upred = gan.generator.predict(udata)        
        '''
        #p.close()
        #p.join()
        print("Total predict time {} secs ".format(time.time()-t1))
        return upred

    def get_gan_slice_serial(self, plane, workers, norm, data=None):
        self.get_norm_constants(workers)
        zid  = int(plane//(self.nzg/self.npz))
        zloc = int(plane%(self.nzg/self.npz))
        gan = PIESRGAN(training_mode=False, height_lr = self.boxsize, \
                    width_lr=self.boxsize, depth_lr = self.boxsize, channels=self.channels, RRDB_layers=RRDB_layers)
        gan.load_weights(generator_weights=self.weights)

        #shared = multiprocessing.Array('f', self.nxg*self.nyg*16*3)
        #upred = np.frombuffer(shared.get_obj(), dtype=np.float32)
        #upred = upred.reshape(self.nxg,self.nyg,16,3)
        udata = np.zeros([self.nxg,self.nyg,self.boxsize,self.channels], dtype=np.float32)
        #ray.init(address='auto', dashboard_host='0.0.0.0', dashboard_port=8888) 
        #p = Pool(workers, initializer = self.init_shared, initargs=(upred,))
        slo = int(max(plane-self.boxsize/2,0))
        shi = int(min(self.boxsize+slo,self.nzg-1))
        if(shi-slo < self.boxsize):
            slo = int(self.boxsize-(shi-slo))

        res=[]
        if(data is None):
            p = Pool(workers)
            for i in range(slo,shi):
                r=p.apply_async(self.get_data_plane, args=(i,), error_callback=self.print_error)
                res.append([r,int(i-slo)])
            for i in range(len(res)):
                udata[:,:,res[i][1],:] = res[i][0].get()
            p.close()
            p.join()

            udata = self.reshape_array([self.boxsize, self.boxsize, self.boxsize], udata)
        else:
            udata = data

        upred = np.zeros([self.nxg, self.nyg, self.boxsize, self.channels], dtype=np.float32)        
        nbatches = int(np.ceil(udata.shape[0]/self.batch_size))
        bx  = self.boxsize
        nby = self.nyg/bx
        if(norm=='minmax'):
            m1=self.mins
            m2=self.maxs
        if(norm=='std'):
            m1=self.mean
            m2=self.std
        if(norm=='range'):
            m1=self.mins
            m2=self.maxs
        
        t2=time.time()
        for i in range(nbatches):
            ist = i*self.batch_size
            ien = min((i+1)*self.batch_size, udata.shape[0])
            t1=time.time()
            data = do_normalisation(udata[ist:ien,:,:,:,:],norm, m1, m2)
            pred = gan.generator.predict(data, batch_size=self.batch_size)
            pred = do_inverse_normalisation(pred, norm, m1, m2)
            print("Predicted in {} secs".format(time.time()-t1))
            for j in range(ist, ien):
                ii = int(j//nby)
                jj = int(j%nby)
                kk=0
                upred[ii*bx:(ii+1)*bx, jj*bx:(jj+1)*bx,kk*bx:(kk+1)*bx,:] = pred[j-ist,:,:,:,:]
 

        print("Total predict time {} secs ".format(time.time()-t2))
        return upred

    def get_gan_filter_slice_serial(self, plane, workers, norm, filter_size, filt):
        self.get_norm_constants(workers)
        zid  = int(plane//(self.nzg/self.npz))
        zloc = int(plane%(self.nzg/self.npz))
        #shared = multiprocessing.Array('f', self.nxg*self.nyg*16*3)
        #upred = np.frombuffer(shared.get_obj(), dtype=np.float32)
        #upred = upred.reshape(self.nxg,self.nyg,16,3)
        zsize = self.boxsize*filter_size
        udata = np.zeros([self.nxg,self.nyg,zsize,self.channels], dtype=np.float32)
        #ray.init(address='auto', dashboard_host='0.0.0.0', dashboard_port=8888) 
        #p = Pool(workers, initializer = self.init_shared, initargs=(upred,))
        slo = int(max(plane-zsize/2,0))
        shi = int(min(zsize+slo,self.nzg-1))
        if(shi-slo < zsize):
            slo = int(zsize-(shi-slo))

        res=[]
    
        p = Pool(workers)
        for i in range(slo,shi):
            r=p.apply_async(self.get_data_plane, args=(i,), error_callback=self.print_error)
            res.append([r,int(i-slo)])
        for i in range(len(res)):
            udata[:,:,res[i][1],:] = res[i][0].get()
        p.close()
        p.join()

        gan = PIESRGAN(training_mode=False, height_lr = self.boxsize, \
                    width_lr=self.boxsize, depth_lr = self.boxsize, channels=self.channels, RRDB_layers=RRDB_layers)
        gan.load_weights(generator_weights=self.weights)


        udata = self.filter_data(filter_size, udata, filt)

        upred = np.zeros([udata.shape[0], udata.shape[1], udata.shape[2], self.channels], dtype=np.float32)        
        udata = self.reshape_array([self.boxsize, self.boxsize, self.boxsize], udata)
        nbatches = int(np.ceil(udata.shape[0]/self.batch_size))
        bx  = self.boxsize
        nby = int(self.nyg/bx/filter_size)
        if(norm=='minmax'):
            m1=self.mins
            m2=self.maxs
        if(norm=='std'):
            m1=self.mean
            m2=self.std
        if(norm=='range'):
            m1=self.mins
            m2=self.maxs
        
        t2=time.time()
        for i in range(nbatches):
            ist = i*self.batch_size
            ien = min((i+1)*self.batch_size, udata.shape[0])
            t1=time.time()
            data = do_normalisation(udata[ist:ien,:,:,:,:],norm, m1, m2)
            pred = gan.generator.predict(data, batch_size=self.batch_size)
            pred = do_inverse_normalisation(pred, norm, m1, m2)
            print("Predicted in {} secs".format(time.time()-t1))
            for j in range(ist, ien):
                ii = int(j//nby)
                jj = int(j%nby)
                kk=0
                upred[ii*bx:(ii+1)*bx, jj*bx:(jj+1)*bx,kk*bx:(kk+1)*bx,:] = pred[j-ist,:,:,:,:]
 

        print("Total predict time {} secs ".format(time.time()-t2))
        return upred


    def pred_gan(self,boxsize, data):
        # Not ideal loading GAN on every proc

        t1=time.time()
        gan = PIESRGAN(training_mode=False, height_lr = self.boxsize, \
                    width_lr=self.boxsize, depth_lr = self.boxsize, channels=self.channels)
        gan.load_weights(generator_weights=self.weights)
        print("Model loaded in {} secs".format(time.time()-t1))
        t1=time.time()
        data = do_normalisation(data,'minmax', self.mins, self.maxs)
        pred = gan.generator.predict(data, batch_size=self.batch_size)
        pred = do_inverse_normalisation(pred, 'minmax', self.mins, self.maxs)
        print("Predicted in {} secs".format(time.time()-t1))
        return pred

    def init_shared(self,shared_arr):
        global upred
        upred = shared_arr

    def print_error(self,err):
        print(err)
        return
