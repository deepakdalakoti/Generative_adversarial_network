# Main training script

''' Use this to train the GAN, run in mode 0 to train only the generator, 
    run in mode 1 to train GAN. Before training GAN, train generator first '''

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

def generate_image_slice(weights, i, norm, DNS, Filt, pref):
    # generate image of a slice of DNS, Filtered data and the predictions
    # Used for monitoring training
    Filt.weights=weights

    pred = Filt.get_gan_slice_serial_upsampling(0,48, norm, 8, data=None)
    HR = DNS.get_data_plane(0)
    LR = Filt.get_data_plane(0)

    Image_generator(LR[:,:,0], pred[:,:,0,0], HR[:,:,0], pref+'_{}.png'.format(i))
    return

def generate_spectrum(weights, i,  norm, Filt, pref):
    # Generate velocity spectrum for predictions
    # Used for monitoring training
    Filt.weights=weights
    spectrum = Filt.get_spectrum_slice(5e-3, 0, 5e-3, 0, 0, norm, 48, pref=pref+"_{}_".format(i))
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

nxg, nyg, nzg = 192, 192, 192
batch_size=16
boxsize=8
upsc = 8
epochs=50000000
print_frequency = 1
save_frequency = 50
fig_frequency = 50

savedir = './data/'
savepref = 'weights'
logdir  = './logs/'
mode = 0 # 1 = GAN, 0 = generator
datapath_train='/scratch/w47/share/IsotropicTurb'
train_loader_hr = DataLoader_s3d(datapath_train+'/DNS_Relambda_162_up_50_Lt_2mm/s-7.0000E-06', nxg*upsc, nyg*upsc, nzg*upsc, 2, batch_size, boxsize*upsc, 1)
train_loader_hr.get_norm_constants(48)
train_loader_lr=GAN_post(datapath_train+'/Filtered_Relambda_162_up_50_Lt_2mm/Filt_8x_regrid_192grid/s-7.0000E-06', nxg, nyg, nzg, 2, batch_size, boxsize, 1, 1)
norm='std'
mins = train_loader_hr.mean
maxs = train_loader_hr.std

if(mode==0):
    ''' Use keras fit API here, seems faster than custom training loop'''
    ''' This mode will only train the generator ''' 
    gan = PIESRGAN(training_mode=True,
                    height_lr = boxsize, width_lr=boxsize, depth_lr=boxsize,
                    gen_lr=4.0e-3, dis_lr=2e-6,
                    channels=1, RRDB_layers=6,
                    upscaling_factor=upsc,
                    )
    #load_model_generator(gan, savedir, 350)
    # callbacks
    #log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tensorboard_stats(5, logdir, 0)
    csv_logger = CSVLogger("upsamp_stats.csv", append=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.95, patience=5, min_lr=0.00005,verbose=1,mode='min')
    save_wghts = write_all_wghts(5, savedir, savepref, 0)

    idx=0

    for i in range(0,epochs):
        imgs_lr = do_normalisation(train_loader_lr.getTrainData(idx),norm, mins, maxs)
        imgs_hr = do_normalisation(train_loader_hr.getTrainData(idx),norm,mins,maxs) 

        idx=idx+1
        if(idx>train_loader_lr.nbatches-1):
            idx=0

        gan.generator.fit(
                    imgs_lr, imgs_hr,
                    batch_size=batch_size,
                    epochs=1,
                    callbacks =  [csv_logger, save_wghts, tensorboard_callback, reduce_lr]
                    )
        if(i%fig_frequency==0):
            if(i==0):
                continue

            generate_image_slice(f'{savedir}{savepref}_{i}.h5', i, 'std',train_loader_hr, train_loader_lr,'test') 
            generate_spectrum(f'{savedir}{savepref}_{i}.h5', i, 'std', train_loader_lr,'test') 

if(mode==1):
    ''' This mode will train generator and discrimitator together '''
    ''' keras API cannot be used directly to train GAN so need to use custom training '''

    logfile = open('GAN_logs','w')
    logfile.write('epoch \t total_loss \t grad_loss\t gen_loss\t pixel_loss\t cont_loss\t dis_loss\n')

    gan = PIESRGAN(training_mode=True,
                    height_lr = boxsize, width_lr=boxsize, depth_lr=boxsize,
                    gen_lr=1e-5, dis_lr=1.0e-5,
                    channels=1, 
                    RRDB_layers=6,
                    upscaling_factor=upsc

                    )
    #gan.build_gan(gen_weights=savedir+'weights_200.h5', disc_weights=None)
    gan.build_gan()
    idx=0
    isave=0
    for i in range(0,epochs):
        imgs_lr = do_normalisation(train_loader_lr.getTrainData(idx),norm, mins, maxs)
        imgs_hr = do_normalisation(train_loader_hr.getTrainData(idx),norm,mins,maxs)
        idx=idx+1
        if(idx>train_loader_lr.nbatches-1):
            idx=0

        t1=time.time()    
        total_loss, grad_loss, gen_loss, pixel_loss, cont_loss, disc_loss = gan.distributed_train_step(imgs_lr, imgs_hr)
        logfile.write("{},{},{},{},{},{},{}\n".format(i, total_loss, grad_loss, gen_loss, pixel_loss, cont_loss, disc_loss))
        logfile.flush()
        if(i%print_frequency == 0):
            print("EPOCH : {} TOTAL GEN LOSS : {} DISC LOSS : {} EPOCH TIME {} secs".format(i, total_loss, disc_loss, time.time()-t1))
        if(i%save_frequency == 0):
            save_model_gan(gan,savedir, i)
            isave=i
        if(i%fig_frequency==0):
            if(i==0):
                continue
            print(isave)
            generate_image_slice(f'{savedir}gen_gan_idx_{isave}.h5',isave, 'std',train_loader_hr, train_loader_lr,'test') 
            generate_spectrum(f'{savedir}gen_gan_idx_{isave}.h5', isave, 'std', train_loader_lr,'test') 

    save_model_gan(gan,savedir, epochs)
    logfile.close()


