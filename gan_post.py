from util import DataLoader_s3d, do_normalisation
from PIESRGAN import PIESRGAN
import multiprocessing 
from multiprocessing import Pool
import numpy as np
import time

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

def init_shared(shared_arr_):
        global ush
        ush = shared_arr_

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
