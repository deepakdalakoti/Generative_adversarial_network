from scipy.io import FortranFile
from scipy.fft import fftn, fft
import time
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftfreq
from .util import DataLoader_s3d
from scipy.stats import binned_statistic
# TODO: Potential speedup by changing loop order according to mem layout
def do_spectrum2(data, xmax, xmin, ymax, ymin, name, channels=3):
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

        tkeh = np.zeros(int(nk))
        del_k = kmax/(nk-1) 

        wavenumbers = np.arange(0,nk)*del_k

        # Take care of fortran vs python indices
        i_g = np.array([i + 1 for i in range(nxg)])
        j_g = np.array([i + 1 for i in range(nyg)])
        kx = np.where(i_g > nhx, -(nxg+1-i_g)*factx, (i_g-1)*factx)
        ky = np.where(j_g > nhy, -(nyg+1-j_g)*facty, (j_g-1)*facty)

        t1 = time.time()
        for i in range(nxg):
                for j in range(nyg):

                        mag_k = np.sqrt(kx[i]**2 + ky[j]**2 )
                        ik = int(min(max(np.floor(mag_k/del_k),0.0),nk-1))
                        #for L in range(channels):
                            #tkeh[ik] = tkeh[ik] + np.real(0.5*(u_k[i,j,L]*np.conj(u_k[i,j,L]))) 
                        tkeh[ik] = tkeh[ik] + 0.5*np.real(np.sum(uk[i,j,:]*np.conj(u_k[i,j,:])))

        print("Loop took {} secs".format(time.time()-t1))
        spectrum = np.zeros([int(nk),2])
        spectrum[:,0]=wavenumbers
        spectrum[:,1]=tkeh/del_k
        np.savetxt(name+'spectrum', spectrum)

        return spectrum

def do_spectrum3(data, xmax, xmin, ymax, ymin, zmax, zmin, name, channels=3):
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
        #mag_k = np.array([kx[i]**2 + ky[j]**2 + kz[k]**2 for (i,j,k) in np.ndindex(nxg, nyg, nzg)] )
        #mag_k = np.array([[[np.sqrt(kx[i]**2 + ky[j]**2 + kz[k]**2) for k in range(nzg)] for j in range(nyg)] for i in range(nxg)])
        #print(mag_k.shape)
        #for i in range(nxg):
        #        for j in range(nyg):
        #            for k in range(nzg):

        #                mag_k[i,j,k] = np.sqrt(kx[i]**2 + ky[j]**2 + kz[k]**2 )
                        #ik = int(min(max(np.floor(mag_k/del_k),0.0),nk-1))
                        #for L in range(channels):
                            #tkeh[ik] = tkeh[ik] + np.real(0.5*(u_k[i,j,k,L]*np.conj(u_k[i,j,k,L]))) 
                        #tkeh[ik] = tkeh[ik] + 0.5*np.real(np.sum(u_k[i,j,k,:]*np.conj(u_k[i,j,k,:]))) 
        print("Done first loop")
        ke = 0.5*np.real(np.multiply(u_k, np.conj(u_k)))
        print(ke.shape)
        print(kx.shape)
        #mag_k = mag_k.ravel()
        #ke =  ke.ravel()
        #print(ke.shape)
        #tkeh, _, _    = binned_statistic(mag_k, ke, statistic = 'sum', bins=nk)
        mag_k_max = np.sqrt(np.max(kx)**2 + np.max(ky)**2 + np.max(kz)**2)

        ix = int(np.ceil(nxg/64))
        iy = int(np.ceil(nyg/64))
        iz = int(np.ceil(nzg/64))
        res = []
        p = Pool(48)
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
                        for L in range(data_loader.channels):
                            tkeh[ik] = tkeh[ik] + np.real(0.5*(u_k[i,j,k,L]*np.conj(u_k[i,j,k,L]))) 

        print("Loop took {} secs".format(time.time()-t1))
        return tkeh/del_k

def do_spectrum_slice(data_loader, u_k, xmax, xmin, ymax, ymin, xid, yid):

        nt = data_loader.nxg*data_loader.nyg
        nhx, nhy = data_loader.nxg/2+1, data_loader.nyg/2+1
        nk = max([nhx, nhy])
        factx = 2.0*np.pi/(xmax-xmin)
        facty = 2.0*np.pi/(ymax-ymin)

        kx_max = (nhx-1)*factx
        ky_max = (nhy-1)*facty

        kmax = np.sqrt(kx_max**2+ky_max**2)

        tkeh = np.zeros(int(nk))
        del_k = kmax/(nk-1) 

        # Take care of fortran vs python indices
        i_g = np.array([xid*data_loader.nx + i + 1 for i in range(data_loader.nx)])
        j_g = np.array([yid*data_loader.ny + i + 1 for i in range(data_loader.ny)])
        kx = np.where(i_g > nhx, -(data_loader.nxg+1-i_g)*factx, (i_g-1)*factx)
        ky = np.where(j_g > nhy, -(data_loader.nyg+1-j_g)*facty, (j_g-1)*facty)
        t1 = time.time()
        for i in range(data_loader.nx):
                for j in range(data_loader.ny):
                        mag_k = np.sqrt(kx[i]**2 + ky[j]**2)
                        ik = int(min(max(np.floor(mag_k/del_k),0.0),nk-1))
                        for L in range(data_loader.channels):
                            tkeh[ik] = tkeh[ik] + np.real(0.5*(u_k[i,j,L]*np.conj(u_k[i,j,L]))) 
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
    u_k = np.empty([data_loader.nxg, data_loader.nyg, data_loader.nzg,data_loader.channels], dtype=complex)
    for L in range(data_loader.channels):
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

def get_velocity_spectrum_slice(data_loader, xmax, xmin, ymax, ymin, plane, workers, pref=''):
    # Get the spectrum box by box as saved in s3d savefile
    # Taken from s3d
    
    data_loader.reset_locs()

    t1 = time.time()
    udata = np.zeros([data_loader.nxg,data_loader.nyg,data_loader.boxsize,data_loader.channels], dtype=np.float32)
    p = Pool(workers)
    slo = int(max(plane-data_loader.boxsize/2,0))
    shi = int(min(data_loader.boxsize+slo,data_loader.nzg-1))
    if(shi-slo < data_loader.boxsize):
        slo = int(data_loader.boxsize-(shi-slo))

    res=[]

    for i in range(slo,shi):
        r=p.apply_async(data_loader.get_data_plane, args=(i,), error_callback=print_error)
        res.append([r,int(i-slo)])
    for i in range(len(res)):
        udata[:,:,res[i][1],:] = res[i][0].get()
    p.close()
    p.join()

    print("Total data load time {} secs".format(time.time()-t1))

    nt = data_loader.nxg*data_loader.nyg
    u_k = np.empty([data_loader.nxg, data_loader.nyg,data_loader.channels], dtype=complex)
    for L in range(data_loader.channels):
        t1 = time.time()
        u_k[:,:,L] = fftn(udata[:,:,plane-slo,L], workers=-1)/nt
        print("FFT finished in {} secs".format(time.time()-t1))
    #print("Total KE after fft ", np.sum(np.real(u_k*np.conj(u_k))))
    nhx, nhy = data_loader.nxg/2+1, data_loader.nyg/2+1
    nk = max([nhx, nhy])
    factx = 2.0*np.pi/(xmax-xmin)
    facty = 2.0*np.pi/(ymax-ymin)

    kx_max = (nhx-1)*factx
    ky_max = (nhy-1)*facty

    kmax = np.sqrt(kx_max**2+ky_max**2)
    del_k = kmax/(nk-1)
    wavenumbers = np.arange(0,nk)*del_k

    global energy 
    energy=np.zeros(int(nk))
    
    p = Pool(workers)
    for yid in range(data_loader.npy):
            for xid in range(data_loader.npx):
                xst, xen = xid*data_loader.nx, (xid+1)*data_loader.nx
                yst, yen = yid*data_loader.ny, (yid+1)*data_loader.ny

                p.apply_async(do_spectrum_slice, args = (data_loader, u_k[xst:xen,yst:yen,:], xmax, xmin, ymax, ymin, xid, yid, ), \
                      callback=log_results, error_callback= print_error)

    p.close()
    p.join()
    spectrum = np.zeros([int(nk),2])
    spectrum[:,0]=wavenumbers
    spectrum[:,1]=energy
    np.savetxt(pref+'spectrum', spectrum)
    return spectrum


if __name__=='__main__':
    #DNS = DataLoader_s3d('/scratch/w47/share/IsotropicTurb/DNS/s-2.4500E-05', 1536, 1536, 1536, 2, 16, 32)
    #spectrum = get_velocity_spectrum(DNS, 5e-3, 0, 5e-3 , 0, 5e-3, 0, 48, 'DNS_s-2.4500E-05_')
    #Filt = DataLoader_s3d('/scratch/w47/share/IsotropicTurb/Filt_8x/filt_s-2.4500E-05', 1536, 1536, 1536, 2, 16, 32)
    #spectrum = get_velocity_spectrum(Filt, 5e-3, 0, 5e-3, 0, 5e-3, 0, 48, 'Filt_8x_s-2.4500E-05_')

    #DNS = DataLoader_s3d('/scratch/w47/dkd561/s3d/data/s-5.0000E-07', 1536, 1536, 1536, 2, 16, 32, 3)
    #spectrum = get_velocity_spectrum(DNS, 5e-3, 0, 5e-3 , 0, 5e-3, 0, 48, 'DNS_all_s-5.0000E-07_')
    #Filt=DataLoader_s3d('/scratch/w47/share/IsotropicTurb/Filtered_Relambda_162_up_50_Lt_2mm/Filt_8x_192grid/s-7.0000E-06', 192, 192, 192, 2, 128, 16,3)
    #DNS=DataLoader_s3d('/scratch/w47/share/IsotropicTurb/LES_Relam162/s-7.0000E-06', 192, 192, 192, 2, 128, 16,3)
    #spectrum = get_velocity_spectrum(DNS, 5e-3, 0, 5e-3 , 0, 5e-3, 0, 48, 'Filt_all_LES_s-7.0000E-06_')
    DNS=DataLoader_s3d('/scratch/w47/share/IsotropicTurb/DNS_Relambda_162_up_50_Lt_2mm/s-7.0000E-06', 1536, 1536, 1536, 2, 128, 16,3)
    udata = DNS.read_parallel(48)
    #Filtered = DNS.filter_data(8,udata,'box')
    Filtered = DNS.smooth_3d(8, udata)
    spectrum = do_spectrum3(Filtered, 5e-3, 0, 5e-3, 0, 5e-3, 0, 'Filt_all_smooth_7e-6_')

    #Filt = DataLoader_s3d('/scratch/w47/share/IsotropicTurb/LES_posteriori_data/s-1.5000E-05', 192, 192, 192, 2, 16, 32, 3)
    #spectrum = get_velocity_spectrum(Filt, 5e-3, 0, 5e-3, 0, 5e-3, 0, 48, 'LES2_s-1.5000E-05_')


    #Filt = DataLoader_s3d('/scratch/w47/share/IsotropicTurb/Filt_8x_192grid/s-1.5000E-05', 192, 192, 192, 2, 16, 32, 3)
    #spectrum = get_velocity_spectrum(Filt, 5e-3, 0, 5e-3, 0, 5e-3, 0, 48, 'LES4_s-1.5000E-05_')



