from scipy.io import FortranFile
from scipy.fft import fftn, fft
import time
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftfreq
from PIESRGAN.utils.util import DataLoader_s3d
from scipy.stats import binned_statistic

# TODO: Potential speedup by changing loop order according to mem layout
'''  Contains routines to calculate velocity spectrum 
 Useful to check the spectrum obtained from the neural network predictions
 Based on the implementation in s3d '''


'''  These functions come in several flavours which differ only slightly
 The first type is where you provide an instance of DataLoader_s3d to the function 
 and it will take care of reading the required data
 These ones are get_velocity_spectrum and get_velocity_spectrum_slice
 The second type is where you provide an array of data as input
 These ones are do_spectrum_2D and do_spectrum_3D '''



def get_velocity_spectrum(data_loader, xmax, xmin, ymax, ymin, zmax, zmin, workers, pref=''):

    # Taken from s3d
    # For full 3D box 
    
    data_loader.reset_locs()

    t1 = time.time()
    u = data_loader.read_parallel(workers)
    print("Total data load time {} secs".format(time.time()-t1))

    spectrum = do_spectrum_3D(u, xmax, xmin, ymax, ymin, zmax, zmin, pref, workers, data_loader.channels)
    return spectrum


def get_velocity_spectrum_slice(data_loader, xmax, xmin, ymax, ymin, plane, workers, pref=''):
    # Taken from s3d
    # For a slice of data, ie only 2D X-Y slice for a give Z plane

    data_loader.reset_locs()

    t1 = time.time()
    udata = data_loader.get_data_plane(plane)
    print("Total data load time {} secs".format(time.time()-t1))

    spectrum = do_spectrum_2D(udata, xmax, xmin, ymax, ymin,  pref, workers, data_loader.channels)

    return spectrum


def do_spectrum_2D(data, xmax, xmin, ymax, ymin, name, workers, channels=3):
        # This is where most of the computations are done
        # Taken from s3d

        nxg = data.shape[0]
        nyg = data.shape[1]
        nt = nxg*nyg
        u_k = np.empty([nxg, nyg,channels], dtype=complex)
        for L in range(channels):
            t1 = time.time()
            u_k[:,:,L] = fftn(data[:,:,L], workers=-1)/nt
            print("FFT finished in {} secs".format(time.time()-t1))

        nhx, nhy = nxg/2+1, nyg/2+1 
        nk = max([nhx, nhy])
        factx = 2.0*np.pi/(xmax-xmin)
        facty = 2.0*np.pi/(ymax-ymin)

        kx_max = (nhx-1)*factx
        ky_max = (nhy-1)*facty

        kmax = np.sqrt(kx_max**2+ky_max**2)

        del_k = kmax/(nk-1) 
        wavenumbers = np.arange(0,nk)*del_k

        # Take care of fortran vs python indices
        i_g = np.array([i + 1 for i in range(nxg)])
        j_g = np.array([i + 1 for i in range(nyg)])
        kx = np.where(i_g > nhx, -(nxg+1-i_g)*factx, (i_g-1)*factx)
        ky = np.where(j_g > nhy, -(nyg+1-j_g)*facty, (j_g-1)*facty)

        mag_k = np.zeros_like(u_k)

        t1 = time.time()
        ke = 0.5*np.real(np.multiply(u_k, np.conj(u_k)))
        mag_k_max = np.sqrt(np.max(kx)**2 + np.max(ky)**2 )

        ix = int(np.ceil(nxg/64))
        iy = int(np.ceil(nyg/64))
        res = []
        p = Pool(workers)
        for i in range(ix):
            for j in range(iy):
                    r=p.apply_async(do_comp_2D, args=(kx[i*64:(i+1)*64], ky[j*64:(j+1)*64],  \
                                  ke[i*64:(i+1)*64,j*64:(j+1)*64], nk, (0, mag_k_max)), error_callback=print_error)
                    res.append(r)

        p.close()
        p.join()


        print("Loop took {} secs".format(time.time()-t1))
        spectrum = np.zeros([int(nk),2])
        for i in range(len(res)):
            spectrum[:,1] = spectrum[:,1] + res[i].get()
        spectrum[:,0]=wavenumbers
        spectrum[:,1]=spectrum[:,1]/del_k
        np.savetxt(name+'spectrum', spectrum)

        return spectrum

def do_spectrum_3D(data, xmax, xmin, ymax, ymin, zmax, zmin, name, workers, channels=3):
        # This is where most of the computations are done
        # Taken from s3d
        nxg = data.shape[0]
        nyg = data.shape[1]
        nzg = data.shape[2]
        nt = nxg*nyg*nzg
        u_k = np.empty([nxg, nyg, nzg, channels], dtype=complex)
        for L in range(channels):
            t1 = time.time()
            u_k[:,:,:,L] = fftn(data[:,:,:,L], workers=-1)/nt
            print("FFT finished in {} secs".format(time.time()-t1))

        nhx, nhy, nhz = nxg/2+1, nyg/2+1, nzg/2+1
        nk = max([nhx, nhy, nhz])
        factx = 2.0*np.pi/(xmax-xmin)
        facty = 2.0*np.pi/(ymax-ymin)
        factz = 2.0*np.pi/(zmax-zmin)

        kx_max = (nhx-1)*factx
        ky_max = (nhy-1)*facty
        kz_max = (nhz-1)*factz

        kmax = np.sqrt(kx_max**2+ky_max**2+kz_max**2)

        tkeh = np.zeros(int(nk))
        del_k = kmax/(nk-1) 

        wavenumbers = np.arange(0,nk)*del_k

        # Take care of fortran vs python indices
        i_g = np.array([i + 1 for i in range(nxg)])
        j_g = np.array([i + 1 for i in range(nyg)])
        k_g = np.array([i + 1 for i in range(nzg)])
        kx = np.where(i_g > nhx, -(nxg+1-i_g)*factx, (i_g-1)*factx)
        ky = np.where(j_g > nhy, -(nyg+1-j_g)*facty, (j_g-1)*facty)
        kz = np.where(k_g > nhz, -(nzg+1-k_g)*factz, (k_g-1)*factz)

        mag_k = np.zeros_like(u_k)

        t1 = time.time()
        ke = 0.5*np.real(np.multiply(u_k, np.conj(u_k)))
        mag_k_max = np.sqrt(np.max(kx)**2 + np.max(ky)**2 + np.max(kz)**2)
       
        ix = int(np.ceil(nxg/64))
        iy = int(np.ceil(nyg/64))
        iz = int(np.ceil(nzg/64))
        res = []
        p = Pool(workers)
        for i in range(ix):
            for j in range(iy):
                for k in range(iz):
                    r=p.apply_async(do_comp, args=(kx[i*64:(i+1)*64], ky[j*64:(j+1)*64], kz[k*64:(k+1)*64], \
                                  ke[i*64:(i+1)*64,j*64:(j+1)*64,k*64:(k+1)*64], nk, (0, mag_k_max)), error_callback=print_error)
                    res.append(r)

        p.close()
        p.join()


        print("Loop took {} secs".format(time.time()-t1))
        spectrum = np.zeros([int(nk),2])
        for i in range(len(res)):
            spectrum[:,1] = spectrum[:,1] + res[i].get()
        spectrum[:,0]=wavenumbers
        spectrum[:,1]=spectrum[:,1]/del_k
        np.savetxt(name+'spectrum', spectrum)

        return spectrum

def do_comp(kx, ky, kz, ke, nk, rng):
        nxg = kx.shape[0]
        nyg = ky.shape[0]
        nzg = kz.shape[0]
        mag_k = np.array([[[np.sqrt(kx[i]**2 + ky[j]**2 + kz[k]**2) for k in range(nzg)] for j in range(nyg)] for i in range(nxg)])
        ke = ke.ravel()
        mag_k = mag_k.ravel()
        tkeh, _, _    = binned_statistic(mag_k, ke, statistic = 'sum', bins=nk, range=rng)
        return tkeh

def do_comp_2D(kx, ky, ke, nk, rng):
        nxg = kx.shape[0]
        nyg = ky.shape[0]
        mag_k = np.array([[np.sqrt(kx[i]**2 + ky[j]**2 ) for j in range(nyg)] for i in range(nxg)])
        ke = ke.ravel()
        mag_k = mag_k.ravel()
        tkeh, _, _    = binned_statistic(mag_k, ke, statistic = 'sum', bins=nk, range=rng)
        return tkeh

def log_results(res):
    global energy
    energy = energy+res
    return 

def print_error(err):
    print("Error: ", err)
    return


