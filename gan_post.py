from util import DataLoader_s3d, do_normalisation
from PIESRGAN import PIESRGAN
import multiprocessing 
from multiprocessing import Pool
import numpy as np
import time
import dill
import os
from util import do_normalisation, do_inverse_normalisation
#import pathos.multiprocessing as pm
# TO DO : add channels to dataloader

class GAN_post(DataLoader_s3d):

    def __init__(self,data_loc, nxg, nyg, nzg, nspec, batch_size, boxsize, channels, weights, a_ref = 347.2):
        super().__init__(data_loc, nxg, nyg, nzg, nspec, batch_size, boxsize, a_ref)
        #self.gan = PIESRGAN(training_mode=False, height_lr = boxsize, \
        #            width_lr=boxsize, depth_lr = boxsize, channels=3)

    def get_pred_parallel(self, workers):
        shared = multiprocessing.Array('f', self.nxg*self.nyg*self.nzg*3)
        upred = np.frombuffer(shared.get_obj(), dtype=np.float32)
        upred = upred.reshape(self.nxg,self.nyg,self.nzg,3)
        t1=time.time()
        res=[]
        p = Pool(workers, initializer = self.init_shared, initargs=(upred,))
        for fnum in range(len(self.flist)):
            data = self.readfile(fnum)
            data = self.reshape_array([self.boxsize, self.boxsize, self.boxsize],data)


            r=p.apply_async(self.pred_gan,args=(self.boxsize,data,),  error_callback=print_error)
            res.append([r,fnum])

        p.close()
        p.join()

        for i in range(len(self.flist)):
            fnum = res[i][1]
            zid = fnum//(self.npx*self.npy)
            yid = (fnum%(self.npx*self.npy))//self.npx
            xid = (fnum%(self.npx*self.npy))%self.npx

            nbx = nx/self.boxsize
            nby = ny/self.boxsize
            nbz = nz/self.boxsize
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

    def get_gan_slice(self, plane, workers):
        _,_  = self.get_norm_constants()
        zid  = int(plane//(self.nzg/self.npz))
        zloc = int(plane%(self.nzg/self.npz))
        #shared = multiprocessing.Array('f', self.nxg*self.nyg*16*3)
        #upred = np.frombuffer(shared.get_obj(), dtype=np.float32)
        #upred = upred.reshape(self.nxg,self.nyg,16,3)
        udata = np.zeros([self.nxg,self.nyg,self.boxsize,3], dtype=np.float32)
        
        #p = Pool(workers, initializer = self.init_shared, initargs=(upred,))
        p = Pool(workers)
        slo = int(max(plane-self.boxsize/2,0))
        shi = int(min(self.boxsize+slo,self.nzg-1))
        if(shi-slo < self.boxsize):
            slo = int(self.boxsize-(shi-slo))

        res=[]

        for i in range(slo,shi):
            r=p.apply_async(self.get_data_plane, args=(i,3,), error_callback=print_error)
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
                error_callback=print_error)
            res.append([r,ist, ien])

        p.close()
        p.join()

        bx  = self.boxsize
        nby = self.nyg/bx
        upred = np.zeros([self.nxg, self.nyg, self.boxsize, 3], dtype=np.float32)        
        for i in range(len(res)):
            pred = res[i][0].get()
            ist  = res[i][1]
            ien  = res[i][2]
            for j in range(ist, ien):
                ii = int(j//nby)
                jj = int(j%nby)
                kk=0
                upred[ii*bx:(ii+1)*bx, jj*bx:(jj+1)*bx,kk*bx:(kk+1)*bx,:] = pred[j-ist,:,:,:,:]

        #p.close()
        #p.join()
        print("Total predict time {} secs ".format(time.time()-t1))
        return upred

    def pred_gan(self,boxsize, data):
        # Not ideal loading GAN on every proc

        t1=time.time()
        gan = PIESRGAN(training_mode=False, height_lr = self.boxsize, \
                    width_lr=self.boxsize, depth_lr = self.boxsize, channels=3)
        gan.load_weights(generator_weights='./data/weights/_generator_idx5120.h5')
        print("Model loaded in {} secs".format(time.time()-t1))
        t1=time.time()
        data = do_normalisation(data,'minmax', self.mins, self.maxs)
        pred = gan.generator.predict(data)
        pred = do_inverse_normalisation(pred, 'minmax', self.mins, self.maxs)
        print("Predicted in {} secs".format(time.time()-t1))
        return pred

    def init_shared(self,shared_arr):
        global upred
        upred = shared_arr

