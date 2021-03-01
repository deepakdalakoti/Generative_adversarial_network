from util import DataLoader_s3d, do_normalisation
from PIESRGAN import PIESRGAN
import multiprocessing 
from multiprocessing import Pool
import numpy as np
import time
import dill
import os
#import pathos.multiprocessing as pm
# TO DO : add channels to dataloader

class GAN_post(DataLoader_s3d):

    def __init__(self,data_loc, nxg, nyg, nzg, nspec, batch_size, boxsize, channels, weights, a_ref = 347.2):
        super().__init__(data_loc, nxg, nyg, nzg, nspec, batch_size, boxsize, a_ref)
        #self.gan = PIESRGAN(training_mode=False, height_lr = boxsize, \
        #            width_lr=boxsize, depth_lr = boxsize, channels=3)

    #def readfile(self, idx):
    #    return super().readfile(idx)

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

            #zid = fnum//(self.npx*self.npy)
            #yid = (fnum%(self.npx*self.npy))//self.npx
            #xid = (fnum%(self.npx*self.npy))%self.npx

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
        zid  = int(plane//(self.nzg/self.npz))
        zloc = int(plane%(self.nzg/self.npz))
        #shared = multiprocessing.Array('f', self.nxg*self.nyg*16*3)
        #upred = np.frombuffer(shared.get_obj(), dtype=np.float32)
        #upred = upred.reshape(self.nxg,self.nyg,16,3)
        udata = np.zeros([self.nxg,self.nyg,16,3], dtype=np.float32)
        
        #p = Pool(workers, initializer = self.init_shared, initargs=(upred,))
        p = Pool(workers)
        slo = max(plane-8,0)
        shi = min(16+slo,self.nzg-1)
        print(slo, shi)
        if(shi-slo < 16):
            slo = 16-(shi-slo)

        res=[]
        print(slo, shi)
        for i in range(slo,shi):
            r=p.apply_async(self.get_data_plane, args=(i,3,), error_callback=print_error)
            res.append([r,int(i-slo)])
        for i in range(len(res)):
            udata[:,:,res[i][1],:] = res[i][0].get()
        udata = self.reshape_array([self.boxsize, self.boxsize, self.boxsize], udata)
        nbatches = udata.shape[0]//32
        res=[]
        for i in range(nbatches):
            ist = i*32
            ien = max((i+1)*32, udata.shape[0])
            r = p.apply_async(self.pred_gan, args=(self.boxsize, udata[ist:ien,:,:,:,:],), \
                error_callback=print_error)
            res.append([r,ist, ien])

        p.close()
        p.join()

        for i in range(len(res)):
            pred = res[i][0].get()
            ist  = res[i][1]
            ien  = res[i][2]
            for j in range(ist, ien):
                ii = int(j//(nby*nbz))
                jj = int((j%(nby*nbz))//(nbz))
                kk = int((j%(nby*nbz))%(nbz))
                udata[ii*bx:(ii+1)*bx, jj*bx:(jj+1)*bx,kk*bx:(kk+1)*bx,:] = pred[i-ist,:,:,:,:]


        return udata

    def pred_gan_slice(self):

        upred = self.reshape_array([self.boxsize, self.boxsize, self.boxsize])

    def pred_gan(self,boxsize, data):
        # Not ideal loading GAN on every proc

        t1=time.time()
        gan = PIESRGAN(training_mode=False, height_lr = boxsize, \
                    width_lr=boxsize, depth_lr = boxsize, channels=3)
        print("Model loaded in {} secs".format(time.time()-t1))
        t1=time.time()
        pred = gan.generator.predict(data)
        print("Predicted in {} secs".format(time.time()-t1))
        return pred

    def pred_gan2(self,fnum, nx, ny, nz, boxsize):
        # Not ideal loading GAN on every proc

        t1 = time.time() 
        data = self.readfile(fnum)
        data = self.reshape_array([boxsize, boxsize, boxsize],data)
        print("Data Loaded in {} secs".format(time.time()-t1))
        t1=time.time()
        gan = PIESRGAN(training_mode=False, height_lr = boxsize, \
                    width_lr=boxsize, depth_lr = boxsize, channels=3)
        print("Model loaded in {} secs".format(time.time()-t1))
        t1=time.time()
        pred = gan.generator.predict(data)
        print("Predicted in {} secs".format(time.time()-t1))
        t1=time.time()
        zid = fnum//(self.npx*self.npy)
        yid = (fnum%(self.npx*self.npy))//self.npx
        xid = (fnum%(self.npx*self.npy))%self.npx

        nbx = nx/boxsize
        nby = ny/boxsize
        nbz = nz/boxsize
        bx = boxsize
        for i in range(data.shape[0]):
            ii = int(i//(nby*nbz))
            jj = int((i%(nby*nbz))//(nbz))
            kk = int((i%(nby*nbz))%(nbz))
            xst = xid*self.nx
            yst = yid*self.ny
            zst = zid*self.nz
            upred[xst+ii*bx:xst+(ii+1)*bx, yst+jj*bx:yst+(jj+1)*bx,zst+kk*bx:zst+(kk+1)*bx,:] = pred[i,:,:,:,:] 
        print("Assigned in {} secs {}".format(time.time()-t1, os.getpid()))
        return

    def init_shared(self,shared_arr):
        global upred
        upred = shared_arr



def get_gan_results_parallel(loader, workers):

        #shared = multiprocessing.Array('f', loader.nxg*loader.nyg*loader.nzg*3)
        #ush = np.frombuffer(shared.get_obj(), dtype=np.float32)
        #ush = ush.reshape(loader.nxg,loader.nyg,loader.nzg,3)

        shared = multiprocessing.Array('f', 4)
        ush = np.frombuffer(shared.get_obj(), dtype=np.float32)
        #ush = ush.reshape(loader.nxg,loader.nyg,loader.nzg,3)


        p = Pool(workers, initializer=loader.init_shared, initargs=(ush,))
        for zid in range(loader.npz):
            for yid in range(loader.npy):
                for xid in range(loader.npx):
                    xst, xen = xid*loader.nx, (xid+1)*loader.nx
                    yst, yen = yid*loader.ny, (yid+1)*loader.ny
                    zst, zen = zid*loader.nz, (zid+1)*loader.nz
                    myid = zid*loader.npx*loader.npy + yid*loader.npx + xid
                    fname = loader.data_loc + '/field.' + "{0:0=6d}".format(myid)
                    idx = loader.flist.index(fname)
                    p.apply_async(pred_single, args = (loader.readfile, idx, xst, \
                            xen, yst, yen, zst, zen,), error_callback= print_error)
        p.close()
        p.join()
        #plt.imshow(ush[0:500,0:500,0,0])
        #plt.savefig('testfig.png')
        #print("HERE", ush[0,0,0,0], ush[1,100,0,0])
        return ush

def pred_single(readfile, idx, xst, xen, yst, yen, zst, zen):
        t1 = time.time()
        gan = PIESRGAN(training_mode=False,
                gen_lr=1e-4, dis_lr=1e-5,
                )
 
        print("Loading GAN took {} secs".format(time.time()-t1))
        t1 = time.time()
        data = readfile(idx)
        print("Reading data took {} secs".format(time.time()-t1))
        t1=time.time()
        ush[xst:xen,yst:yen, zst:zen,:] = gan.generator.predict(data[None,:,:,:,:])
        #ush[0,0,0,0] = idx
        #print(ush[0,0,0,0])
        print("Predicted file in {} secs".format(time.time()-t1))
        return

def print_error(err):
    print(err)
    return
def pred_gan(readfile,fnum, u, nx, ny, nz, boxsize):
        # This is not ideal way to do pred in parallel but multiprocessing cant pickle class
        # Maybe some other library does
        # Fine for now
        # Also not ideal loading GAN on every proc
        '''
        data = readfile(fnum)
        data = reshape_array([boxsize, boxsize, boxsize],data)
        #pred = self.gan.predict(data)
        nbx = nx/boxsize
        nby = ny/boxsize
        nbz = nz/boxsize
        bx = boxsize
        for i in range(pred.shape[0]):
            ii = i//(nby*nbz)
            jj = (i%(nby*nbz))//(nbz)
            kk = (i%(nby*nbz))%(nbz)
            u[ii*bx:(ii+1)*bx, jj*bx:(jj+1)*bx,kk*bx:(kk+1)*bx,:] = data[i,:,:,:,:] 
        '''
        print("HERE")
        return



