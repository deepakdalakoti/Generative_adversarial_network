from PIESRGAN import PIESRGAN
from util import DataLoader_s3d, do_normalisation, Image_generator2, do_inverse_normalisation
from util import Image_generator
import time
import tensorflow as tf
import numpy as np
from gan_post import GAN_post
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau
import pickle
#import horovod.tensorflow.keras as hvd

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


nxg, nyg, nzg =192, 192, 192
batch_size=16
boxsize=32
datapath_train='/scratch/w47/share/IsotropicTurb'
train_loader_hr = DataLoader_s3d(datapath_train+'/Filtered_Relambda_162_up_50_Lt_2mm/DNS_regrid_192grid/s-7.0000E-06', nxg, nyg, nzg, 2, batch_size, boxsize, 3)
train_loader_hr.get_norm_constants(48)

train_loader_lr=GAN_post(datapath_train+'/Filtered_Relambda_162_up_50_Lt_2mm/Filt_8x_regrid_192grid/s-7.0000E-06', nxg, nyg, nzg, 2, batch_size, boxsize, 3, 1)

#idx=0
#imgs_lr = do_normalisation(train_loader_lr.getTrainData_plane(48,0,idx),'minmax', mins, maxs)
#imgs_hr = do_normalisation(train_loader_hr.getTrainData_plane(48,0,idx),'minmax',mins,maxs)

epochs=50000000
print_frequency = 200
save_frequency = 50
fig_frequency = 50
savedir = './data/regridded/'
#mirrored_strategy = tf.distribute.MirroredStrategy()
mode = 0 # 1 = GAN, 0 = generator
norm='minmax'
mins = train_loader_hr.mins
maxs = train_loader_hr.maxs


if(mode==1):
    logfile = open('GAN_logs_32_l12','w')
    logfile.write('epoch \t total_loss \t grad_loss\t gen_loss\t pixel_loss\t cont_loss\t dis_loss\n')
    gan = PIESRGAN(training_mode=True,
                    height_lr = boxsize, width_lr=boxsize, depth_lr=boxsize,
                    gen_lr=1e-5, dis_lr=1.0e-5,
                    channels=3, 
                    RRDB_layers=12
                    )


    gan.build_gan(gen_weights=savedir+'generator_idx_5700.h5', disc_weights=None)
    gan.build_gan()
    #load_model_gan(gan, savedir, 5700, Gen=True, Dis=True) 

    #out = gan.distributed_train_step(imgs_lr, imgs_hr)
    idx=0
    isave=0
    for i in range(0,epochs):
        imgs_lr = do_normalisation(train_loader_lr.getData(idx),norm, mins, maxs)
        imgs_hr = do_normalisation(train_loader_hr.getData(idx),norm,mins,maxs)
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
            generate_image_slice(savedir+"gen_gan_idx_{}.h5".format(isave),isave, norm, train_loader_hr, train_loader_lr, 'slice_gan_8_l12')
            generate_spectrum(savedir+"gen_gan_idx_{}.h5".format(isave),isave, norm, train_loader_lr, 'pred_gan_s-7.0000E-06_8_l12')
        #    pred = gan.gen_gan(imgs_lr)
        #    print(np.sum((pred-imgs_hr)**2))
        #    generate_image(pred, imgs_lr, imgs_hr, mins, maxs, i)


    save_model_gan(gan,savedir, epochs)
    logfile.close()

if(mode==0):
    gan = PIESRGAN(training_mode=True,
                    height_lr = boxsize, width_lr=boxsize, depth_lr=boxsize,
                    gen_lr=5e-5, dis_lr=2e-6,
                    channels=3, RRDB_layers=6
                    )
    load_model_generator(gan, savedir, 2800)
    #gan.generator.load_weights(savedir+'generator_idx_11000.h5')
    #gan.generator = tf.keras.models.load_model(savedir+'generator_idx_100')
    csv_logger = CSVLogger("regridded.csv", append=True)
    #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.75, patience=5, min_lr=1e-6, verbose=1)
    #hvd_callback = hvd.callbacks.BroadcastGlobalVariablesCallback(0)
    idx=0
    isave=700
    for i in range(2801,epochs):
        imgs_lr = do_normalisation(train_loader_lr.getData(idx),norm, mins, maxs)
        imgs_hr = do_normalisation(train_loader_hr.getData(idx),norm,mins,maxs)
        idx=idx+1
        if(idx>train_loader_lr.nbatches-1):
            idx=0
        t1=time.time()
        gan.generator.fit(
                    imgs_lr, imgs_hr,
                    steps_per_epoch=1,
                    epochs=1,
                    callbacks = [csv_logger],
                    )
        if(i%print_frequency==0):
            print("Epoch {} out of nbatches {}  took {} secs".format(i,train_loader_lr.nbatches,time.time()-t1))
        if(i%save_frequency == 0):
            #gan.generator.save_weights(savedir+"generator_idx_{}.h5".format(i))
            save_model_generator(gan, savedir, i)
            #gan.generator.save(savedir+"generator_idx_{}".format(i))
            isave=i
        if(i%fig_frequency==0):
            if(i==0):
                continue
            generate_image_slice(savedir+"generator_idx_{}.h5".format(isave),isave, norm, train_loader_hr, train_loader_lr, 'regredded')
            #generate_spectrum(savedir+"generator_idx_{}.h5".format(isave),isave, norm, train_loader_lr, 'pred_regredded')
        #    pred = gan.gen_gan(imgs_lr)
        #    print(np.sum((pred-imgs_hr)**2))
        #    generate_image(pred, imgs_lr, imgs_hr, mins, maxs, i)

    #gan.generator.save_weights(savedir+"generator_idx_{}.h5".format(epochs))
    save_model_generator(gan, savedir, epochs)




