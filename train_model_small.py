from model.PIESRGAN import PIESRGAN
from utils.util import DataLoader_s3d, do_normalisation, Image_generator2, do_inverse_normalisation
from utils.util import Image_generator, write_all_wghts, tensorboard_stats
import time
import tensorflow as tf
import numpy as np
from utils.gan_post import GAN_post
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau
import pickle
import datetime
#import horovod.tensorflow.keras as hvd
tf.random.set_seed(0)
def generate_image(pred, imgs_lr, imgs_hr, mins, maxs, norm, idx):

    pred = do_inverse_normalisation(pred, norm, mins, maxs)
    pred = train_loader_hr.inverse_reshape_array(pred)
    imgs_lr = train_loader_hr.inverse_reshape_array(imgs_lr)
    imgs_lr = do_inverse_normalisation(imgs_lr, norm, mins, maxs)
    imgs_hr = train_loader_hr.inverse_reshape_array(imgs_hr)
    imgs_hr = do_inverse_normalisation(imgs_hr, norm, mins, maxs)
    Image_generator2(imgs_lr[:,:,0,0], pred[:,:,0,0], imgs_hr[:,:,0,0], 'testfig_{}.png'.format(idx))

def generate_image_slice(weights, i, norm, DNS, Filt, pref):
     
    Filt.weights=weights

    pred = Filt.get_gan_slice_serial(0,48, norm, data=None)
    HR = DNS.get_data_plane(0)

    #Filt=DataLoader_s3d(datapath_train+'/Filt_4x/filt_s-1.5000E-05', 1536, 1536, 1536, 2, batch_size, boxsize, 1)
    LR = Filt.get_data_plane(0)

    Image_generator(LR[:,:,0], pred[:,:,0,0], HR[:,:,0], pref+'_{}.png'.format(i))
    return

def generate_spectrum(weights, i,  norm, Filt, pref):
    Filt.weights=weights
    spectrum = Filt.get_spectrum_slice(48, 5e-3, 0, 5e-3, 0, 0, norm, savePred=None, pref=pref+"_{}_".format(i))
    return

def save_model_generator(model, savedir, idx):
    model.generator.save_weights(savedir+'generator_idx_{}.h5'.format(idx))
    opt_wght = model.optimizer.get_weights()
    pickle.dump(opt_wght,open(savedir+'generator_opt_idx_{}.h5'.format(idx),'wb'))
    return

def load_model_generator(model,savedir, idx):
    print("loading model from epoch {}".format(idx))
    model.generator.load_weights(savedir+'test_{}.h5'.format(idx))
    opt_wght = pickle.load(open(savedir+'test_opt_idx_{}.h5'.format(idx),'rb'))
    def load_opt():
        grad_vars = model.generator.trainable_weights
        zero_grads = [tf.zeros_like(w) for w in grad_vars]
        model.optimizer.apply_gradients(zip(zero_grads, grad_vars)) 
        model.optimizer.set_weights(opt_wght)
    model.mirrored_strategy.run(load_opt)
    return

def save_model_gan(model, savedir, idx):
    model.gen_gan.save_weights(savedir+'gen_gan_idx_{}.h5'.format(idx))
    opt_wght = model.generator_optimizer.get_weights()
    pickle.dump(opt_wght,open(savedir+'gen_gan_opt_idx_{}.h5'.format(idx),'wb'))

    model.dis_gan.save_weights(savedir+'dis_gan_idx_{}.h5'.format(idx))
    opt_wght = model.discriminator_optimizer.get_weights()
    pickle.dump(opt_wght,open(savedir+'dis_gan_opt_idx_{}.h5'.format(idx),'wb'))
    return

def load_model_gan(model,savedir, idx, Gen=True, Dis=False):
    print("loading model from epoch {}".format(idx))
    if(Gen):
        model.gen_gan.load_weights(savedir+'gen_gan_idx_{}.h5'.format(idx))
        opt_wght = pickle.load(open(savedir+'gen_gan_opt_idx_{}.h5'.format(idx),'rb'))
        def load_opt():
            grad_vars = model.gen_gan.trainable_weights
            zero_grads = [tf.zeros_like(w) for w in grad_vars]
            model.generator_optimizer.apply_gradients(zip(zero_grads, grad_vars)) 
            model.generator_optimizer.set_weights(opt_wght)
        model.mirrored_strategy.run(load_opt)
    if(Dis):
        model.dis_gan.load_weights(savedir+'dis_gan_idx_{}.h5'.format(idx))
        opt_wght = pickle.load(open(savedir+'dis_gan_opt_idx_{}.h5'.format(idx),'rb'))

        def load_opt_dis():
            grad_vars = model.dis_gan.trainable_weights
            zero_grads = [tf.zeros_like(w) for w in grad_vars]
            model.discriminator_optimizer.apply_gradients(zip(zero_grads, grad_vars)) 
            model.discriminator_optimizer.set_weights(opt_wght)
        model.mirrored_strategy.run(load_opt_dis)

    return

nxg, nyg, nzg = 384, 384, 384
batch_size=128
boxsize=32
upsc = 4
epochs=50000000
print_frequency = 200
save_frequency = 500
fig_frequency = 50
savedir = './data/'
#mirrored_strategy = tf.distribute.MirroredStrategy()
mode = 0 # 1 = GAN, 0 = generator
datapath_train='/scratch/w47/share/IsotropicTurb'
train_loader_hr = DataLoader_s3d(datapath_train+'/DNS_Relambda_162_up_50_Lt_2mm/s-7.0000E-06', nxg*upsc, nyg*upsc, nzg*upsc, 2, batch_size, boxsize*upsc, 1)
train_loader_hr.get_norm_constants(48)
train_loader_lr=GAN_post(datapath_train+'/Filtered_Relambda_162_up_50_Lt_2mm/Filt_4x_regrid_384/s-7.0000E-06', nxg, nyg, nzg, 2, batch_size, boxsize, 1, 1)
norm='std'
mins = train_loader_hr.mean
maxs = train_loader_hr.std



udns = train_loader_hr.read_parallel(48)
ufilt = train_loader_lr.read_parallel(48)
udns = train_loader_hr.reshape_array([boxsize*upsc, boxsize*upsc, boxsize*upsc], udns)
ufilt = train_loader_lr.reshape_array([boxsize, boxsize, boxsize], ufilt)


if(mode==0):
    ''' Use keras fit API here, seems faster than custom training loop'''
 
    gan = PIESRGAN(training_mode=True,
                    height_lr = boxsize, width_lr=boxsize, depth_lr=boxsize,
                    gen_lr=4.0e-3, dis_lr=2e-6,
                    channels=1, RRDB_layers=6,
                    upscaling_factor=upsc,
                    )
    #load_model_generator(gan, savedir, 350)
    # callbacks
    #log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tensorboard_stats(5, './logs/upsamp2/', 0)
    csv_logger = CSVLogger("upsamp_2.csv", append=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.95, patience=5, min_lr=0.00005,verbose=1,mode='min')
    save_wghts = write_all_wghts(5, './data/upsamp2/', 'test', 0)

    idx=0

    #rands = np.arange(udns.shape[0])
    #np.random.shuffle(rands)
    #udns = udns[rands,:,:,:,:]
    #ufilt = ufilt[rands,:,:,:,:]
    ufilt = do_normalisation(ufilt,norm, mins, maxs)
    udns = do_normalisation(udns,norm, mins, maxs)
        
    for i in range(0,epochs):
 
        #imgs_hr = udns[idx*batch_size:(idx+1)*batch_size, :,:,:,:]
        #imgs_lr = ufilt[idx*batch_size:(idx+1)*batch_size,:,:,:,:]

        idx=idx+1
        if(idx>train_loader_lr.nbatches-1):
            idx=0

        #t1=time.time()
        gan.generator.fit(
                    ufilt, udns,
                    batch_size=24,
                    epochs=5000,
                    #callbacks = callbacks,
                    #callbacks = [csv_logger, tensorboard_callback, reduce_lr, check],
                    callbacks =  [csv_logger, save_wghts, tensorboard_callback, reduce_lr]
                    )
        #print("TIME {}".format(time.time()-t1))

        #for j in range(1):
        #    loss=gan.distributed_train_gen(imgs_lr, imgs_hr)
        #    logfile.write("{},{}\n".format(i,loss))
        #    logfile.flush()
        #if(i%print_frequency==0):
        #    print("Epoch {} out of nbatches {}  took {} secs LOSS {}".format(i,train_loader_lr.nbatches,time.time()-t1, loss))




