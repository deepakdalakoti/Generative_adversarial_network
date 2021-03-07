from scipy.io import FortranFile
from scipy.fft import fftn, fft
import time
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftfreq
from util import DataLoader_s3d
# TODO: Potential speedup by changing loop order according to mem layout
def do_spectrum(data_loader, u_k, xmax, xmin, ymax, ymin, zmax, zmin, xid, yid, zid):

        nt = data_loader.nxg*data_loader.nyg*data_loader.nzg
        nhx, nhy, nhz = data_loader.nxg/2+1, data_loader.nyg/2+1, data_loader.nzg/2+1
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

        # Take care of fortran vs python indices
        i_g = np.array([xid*data_loader.nx + i + 1 for i in range(data_loader.nx)])
        j_g = np.array([yid*data_loader.ny + i + 1 for i in range(data_loader.ny)])
        k_g = np.array([zid*data_loader.nz + i + 1 for i in range(data_loader.nz)])
        kx = np.where(i_g > nhx, -(data_loader.nxg+1-i_g)*factx, (i_g-1)*factx)
        ky = np.where(j_g > nhy, -(data_loader.nyg+1-j_g)*facty, (j_g-1)*facty)
        kz = np.where(k_g > nhz, -(data_loader.nzg+1-k_g)*factz, (k_g-1)*factz)

        #kx = fftfreq(data_loader.nxg, (xmax-xmin)/data_loader.nxg)*2.0*np.pi
        #ky = fftfreq(data_loader.nyg, (ymax-ymin)/data_loader.nyg)*2.0*np.pi
        #kz = fftfreq(data_loader.nzg, (zmax-zmin)/data_loader.nzg)*2.0*np.pi
        t1 = time.time()
        for i in range(data_loader.nx):
                for j in range(data_loader.ny):
                    for k in range(data_loader.nz):

                        mag_k = np.sqrt(kx[i]**2 + ky[j]**2 + kz[k]**2)
                        ik = int(min(max(np.floor(mag_k/del_k),0.0),nk-1))
                        tkeh[ik] = tkeh[ik] + np.real(0.5*(u_k[i,j,k,0]*np.conj(u_k[i,j,k,0]) + \
                                   u_k[i,j,k,1]*np.conj(u_k[i,j,k,1]) + \
                                   u_k[i,j,k,2]*np.conj(u_k[i,j,k,2])))
        print("Loop took {} secs".format(time.time()-t1))
        return tkeh/del_k

def log_results(res):
    global energy
    energy = energy+res
    return 

def print_error(err):
    print("Fucked", err)
    return

def get_velocity_spectrum(data_loader, xmax, xmin, ymax, ymin, zmax, zmin, workers, pref=''):
    # Get the spectrum box by box as saved in s3d savefile
    # Taken from s3d
    
    data_loader.reset_locs()

    t1 = time.time()
    u = data_loader.read_parallel(workers)
    print("Total data load time {} secs".format(time.time()-t1))

    nt = data_loader.nxg*data_loader.nyg*data_loader.nzg
    u_k = np.empty([data_loader.nxg, data_loader.nyg, data_loader.nzg,3], dtype=complex)
    for L in range(3):
        t1 = time.time()
        u_k[:,:,:,L] = fftn(u[:,:,:,L], workers=-1)/nt
        print("FFT finished in {} secs".format(time.time()-t1))
    #print("Total KE after fft ", np.sum(np.real(u_k*np.conj(u_k))))
    nhx, nhy, nhz = data_loader.nxg/2+1, data_loader.nyg/2+1, data_loader.nzg/2+1
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

    global energy 
    energy=np.zeros(int(nk))
    
    p = Pool(workers)
    for zid in range(data_loader.npz):
        for yid in range(data_loader.npy):
            for xid in range(data_loader.npx):
                xst, xen = xid*data_loader.nx, (xid+1)*data_loader.nx
                yst, yen = yid*data_loader.ny, (yid+1)*data_loader.ny
                zst, zen = zid*data_loader.nz, (zid+1)*data_loader.nz

                p.apply_async(do_spectrum, args = (data_loader, u_k[xst:xen,yst:yen,zst:zen,:], xmax, xmin, ymax, ymin, zmax, zmin,xid, yid, zid,), \
                      callback=log_results, error_callback= print_error)

    p.close()
    p.join()
    spectrum = np.zeros([int(nk),2])
    spectrum[:,0]=wavenumbers
    spectrum[:,1]=energy
    np.savetxt(pref+'spectrum', spectrum)
    return spectrum

if __name__=='__main__':
    DNS = DataLoader_s3d('/scratch/w47/share/IsotropicTurb/DNS/s-2.4500E-05', 1536, 1536, 1536, 2, 16, 32)
    spectrum = get_velocity_spectrum(DNS, 5e-3, 0, 5e-3 , 0, 5e-3, 0, 48, 'DNS_s-2.4500E-05_')
    Filt = DataLoader_s3d('/scratch/w47/share/IsotropicTurb/Filt_8x/filt_s-2.4500E-05', 1536, 1536, 1536, 2, 16, 32)
    spectrum = get_velocity_spectrum(Filt, 5e-3, 0, 5e-3, 0, 5e-3, 0, 48, 'Filt_8x_s-2.4500E-05_')

    DNS = DataLoader_s3d('/scratch/w47/share/IsotropicTurb/DNS/s-1.5000E-05', 1536, 1536, 1536, 2, 16, 32)
    spectrum = get_velocity_spectrum(DNS, 5e-3, 0, 5e-3 , 0, 5e-3, 0, 48, 'DNS_s-1.5000E-05_')
    Filt = DataLoader_s3d('/scratch/w47/share/IsotropicTurb/Filt_8x/filt_s-1.5000E-05', 1536, 1536, 1536, 2, 16, 32
    spectrum = get_velocity_spectrum(Filt, 5e-3, 0, 5e-3, 0, 5e-3, 0, 48, 'Filt_8x_s-1.5000E-05_')

