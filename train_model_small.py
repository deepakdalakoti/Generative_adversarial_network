from model.PIESRGAN import PIESRGAN
from utils.util import DataLoader_s3d, do_normalisation, Image_generator2, do_inverse_normalisation
from utils.util import Image_generator
import time
import tensorflow as tf
import horovod.tensorflow.keras as hvd
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
    model.generator.load_weights(savedir+'generator_idx_{}.h5'.format(idx))
    opt_wght = pickle.load(open(savedir+'generator_opt_idx_{}.h5'.format(idx),'rb'))
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

hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

opt = tf.keras.optimizers.Adam(learning_rate=5e-4)
distopt = hvd.DistributedOptimizer(opt)
nxg, nyg, nzg = 384, 384, 384
batch_size=96
boxsize=32
upsc = 4
#del udns
#del ufilt
if(mode==0):
    ''' Use keras fit API here, seems faster than custom training loop'''
 
    gan = PIESRGAN(training_mode=True,
                    height_lr = boxsize, width_lr=boxsize, depth_lr=boxsize,
                    gen_lr=5.0e-4, dis_lr=2e-6,
                    channels=1, RRDB_layers=3,
                    upscaling_factor=upsc,
                    opt = distopt
                    )
    gan.batch_size=batch_size
    callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    ] 
    #load_model_generator(gan, savedir, 0)
    # callbacks
    #log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = tensorboard_stats(50, './logs/', 0)
    #csv_logger = CSVLogger("upsamp.csv", append=True)
    #reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.95, patience=5, min_lr=0.00001,verbose=1,mode='min')
    #save_wghts = write_all_wghts(100, './data/', 'test', 0)

    idx=0

    rands = np.arange(udns.shape[0])
    np.random.shuffle(rands)
    udns = udns[rands,:,:,:,:]
    ufilt = ufilt[rands,:,:,:,:]
    ufilt = do_normalisation(ufilt,norm, mins, maxs)
    udns = do_normalisation(udns,norm, mins, maxs)
        
    for i in range(0,epochs):
 
        imgs_hr = udns[idx*batch_size:(idx+1)*batch_size, :,:,:,:]
        imgs_lr = ufilt[idx*batch_size:(idx+1)*batch_size,:,:,:,:]

        idx=idx+1
        if(idx>train_loader_lr.nbatches-1):
            idx=0

        #t1=time.time()
        gan.generator.fit(
                    imgs_lr, imgs_hr,
                    batch_size=8,
                    epochs=10,
                    callbacks = callbacks,
                    verbose=1 if hvd.rank() == 0 else 0
                    #callbacks = [csv_logger, tensorboard_callback, reduce_lr, check],
                    #callbacks =  [csv_logger, save_wghts, tensorboard_callback, reduce_lr]
                    )
        #print("TIME {}".format(time.time()-t1))

        #for j in range(1):
        #    loss=gan.distributed_train_gen(imgs_lr, imgs_hr)
        #    logfile.write("{},{}\n".format(i,loss))
        #    logfile.flush()
        #if(i%print_frequency==0):
        #    print("Epoch {} out of nbatches {}  took {} secs LOSS {}".format(i,train_loader_lr.nbatches,time.time()-t1, loss))




