#! /usr/bin/python
# -*- coding: utf-8 -*-
# ! /usr/bin/python
import os          # enables interactions with the operating system
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
import sys
import pickle      # object-->byte system
import datetime    # manipulating dates and times
import numpy as np

# Import keras + tensorflow without the "Using XX Backend" message
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import gc
import tensorflow as tf
#import tensorlayer as tl
#from tensorlayer.layers import Conv3dLayer, LambdaLayer
from keras.utils import plot_model
from keras import layers
from keras import Model
from keras.models import Model, load_model
from keras.layers import Input, Activation, Add, Concatenate, Multiply
from keras.layers import BatchNormalization, LeakyReLU, PReLU, Conv2D, Dense, Conv3D,ZeroPadding3D
from keras.layers import UpSampling2D, Lambda, Dropout
from keras.optimizers import Adam, RMSprop
from keras.applications.vgg19 import preprocess_input
from keras.utils.data_utils import OrderedEnqueuer
from keras import backend as K           #campatability
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
from keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import CSVLogger
sys.stderr = stderr
from util import UpSampling3D, DataLoader, RandomLoader_train, Image_generator 
#subPixelConv3d, DataLoader, RandomLoader_train, Image_generator
from util import DataLoader_s3d, do_normalisation
import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
'''Use Horovod in case of multi nodes parallelizations'''
#import horovod.keras as hvd
import time
import logging
def lrelu1(x):
    return tf.maximum(x, 0.25 * x)
def lrelu2(x):
    return tf.maximum(x, 0.3 * x)

def grad0(matrix): 
    return np.gradient(matrix,axis=0)

def grad1(matrix): 
    return np.gradient(matrix,axis=1)

def grad2(matrix): 
    return np.gradient(matrix,axis=2)

def grad3(matrix): 
    return np.gradient(matrix,axis=3)

class PIESRGAN():
    """
    Implementation of PIESRGAN as described in the paper
    """

    def __init__(self,
                 height_lr=16, width_lr=16, depth_lr=16,
                 gen_lr=1e-4, dis_lr=1e-7,
                 channels=3,
                 # VGG scaled with 1/12.75 as in paper
                 loss_weights={'percept':1e-1,'gen':5e-5, 'pixel':5e0},
                 training_mode=True,
                 refer_model=None,
                 ):
        """
        :param int height_lr: Height of low-resolution DNS data
        :param int width_lr: Width of low-resolution DNS data
        :param int depth: Width of low-resolution DNS data
        :param int upscaling_factor: Up-scaling factor
        :param int gen_lr: Learning rate of generator
        :param int dis_lr: Learning rate of discriminator
        """

        self.mirrored_strategy = tf.distribute.MirroredStrategy()
        self.upscaling_factor=1.0
        # Low-resolution image dimensions
        self.height_lr = height_lr
        self.width_lr = width_lr
        self.depth_lr=depth_lr
        self.channels_lr = int(channels)
        # High-resolution image dimensions are identical to those of the LR, removed the upsampling block!
        self.height_hr = int(self.height_lr )
        self.width_hr = int(self.width_lr )
        self.depth_hr = int(self.depth_lr )
        self.channels_hr = int(channels)
        # Low-resolution and high-resolution shapes
        """ DNS-Data only has one channel, when only using PS field, when using u,v,w,ps, change to 4 channels """
        self.shape_lr = (self.height_lr, self.width_lr, self.depth_lr,self.channels_lr)
        self.shape_hr = (self.height_hr, self.width_hr, self.depth_hr,self.channels_hr)
        self.batch_shape_lr = (None,self.height_lr, self.width_lr, self.depth_lr,self.channels_lr)
        self.batch_shape_hr = (None,self.height_hr, self.width_hr, self.depth_hr,self.channels_hr)
        # Learning rates
        self.gen_lr = gen_lr
        self.dis_lr = dis_lr
        
        # Scaling of losses
        self.loss_weights = loss_weights
        
        # Gan setup settings
        self.gan_loss = 'mse'
        self.dis_loss = 'binary_crossentropy'
        
        # Build & compile the generator network
        with self.mirrored_strategy.scope():
            self.generator = self.build_generator()
            self.compile_generator(self.generator)
            #self.refer_model = refer_model
            # If training, build rest of GAN network
            if training_mode:
                self.discriminator = self.build_discriminator()
                self.RaGAN = self.build_RaGAN()
                self.piesrgan = self.build_piesrgan()
                #self.compile_discriminator(self.RaGAN)
                #self.compile_piesrgan(self.piesrgan)
                
    def save_weights(self, filepath, e=None):
        """Save the generator and discriminator networks"""
        self.generator.save_weights("{}_generator_{}X_epoch{}.h5".format(filepath, self.upscaling_factor, e))
        self.discriminator.save_weights("{}_discriminator_{}X_epoch{}.h5".format(filepath, self.upscaling_factor, e))
        

    def load_weights(self, generator_weights=None, discriminator_weights=None, **kwargs):
        if generator_weights:
            self.generator.load_weights(generator_weights, **kwargs)
        if discriminator_weights:
            self.discriminator.load_weights(discriminator_weights, **kwargs)
            
            
                
    def build_generator(self, ):
        """
         Build the generator network according to description in the paper.
         First define seperate blocks then assembly them together
        :return: the compiled model
        """
        w_init = tf.random_normal_initializer(stddev=0.02)
        height_hr=self.height_hr
        width_hr=self.width_hr
        depth_hr=self.depth_hr

        def dense_block(input):
            x1 = Conv3D(64, kernel_size=3, strides=1, padding='same')(input)
            x1 = LeakyReLU(0.2)(x1)
            x1 = Concatenate()([input, x1])

            x2 = Conv3D(64, kernel_size=3, strides=1, padding='same')(x1)
            x2 = LeakyReLU(0.2)(x2)
            x2 = Concatenate()([input, x1, x2])

            x3 = Conv3D(64, kernel_size=3, strides=1, padding='same')(x2)
            x3 = LeakyReLU(0.2)(x3)
            x3 = Concatenate()([input, x1, x2, x3])

            x4 = Conv3D(64, kernel_size=3, strides=1, padding='same')(x3)
            x4 = LeakyReLU(0.2)(x4)
            x4 = Concatenate()([input, x1, x2, x3, x4])  #added x3, which ESRGAN didn't include

            x5 = Conv3D(64, kernel_size=3, strides=1, padding='same')(x4)
            x5 = Lambda(lambda x: x * 0.2)(x5)
            """here: assumed beta=0.2"""
            x = Add()([x5, input])
            return x

        def RRDB(input):
            x = dense_block(input)
            x = dense_block(x)
            x = dense_block(x)
            """here: assumed beta=0.2 as well"""
            x = Lambda(lambda x: x * 0.2)(x)
            out = Add()([x, input])
            return out

        """----------------Assembly the generator-----------------"""
        # Input low resolution image
        #lr_input = Input(shape=(None, None, None,3))
        #with self.mirrored_strategy.scope():
        lr_input = Input(shape=self.shape_lr)
        # Pre-residual
        x_start = Conv3D(64, data_format="channels_last", kernel_size=3, strides=1, padding='same')(lr_input)
        x_start = LeakyReLU(0.2)(x_start)

        # Residual-in-Residual Dense Block
        x = RRDB(x_start)

        # Post-residual block
        x = Conv3D(64, kernel_size=3, strides=1, padding='same')(x)
        x = Lambda(lambda x: x * 0.2)(x)
        x = Add()([x, x_start])
        x = Conv3D(512,kernel_size=3, strides=1, padding='same',activation=lrelu1)(x)
        x = Conv3D(512,kernel_size=3, strides=1, padding='same',activation=lrelu1)(x)
        #Final 2 convolutional layers
        x = Conv3D(64, kernel_size=3, strides=1, padding='same')(x)
        x = LeakyReLU(0.2)(x)
        hr_output = Conv3D(self.channels_hr, kernel_size=3, strides=1, padding='same', activation='tanh')(x)

        # Create model and compile
        model = Model(inputs=lr_input, outputs=hr_output)
        #Uncomment this if using multi GPU model
        #model=multi_gpu_model(model,gpus=2,cpu_merge=True)
        # model.summary()
        return model
        
        
        
    def build_discriminator(self, filters=64):
        """
        Build the discriminator network according to description in the paper.
        :param optimizer: Keras optimizer to use for network
        :param int filters: How many filters to use in first conv layer
        :return: the compiled model
        """

        def conv3d_block(input, filters, strides=1, bn=True):
            d = Conv3D(filters, kernel_size=3, strides=strides, padding='same')(input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        #with self.mirrored_strategy.scope():
        # Input high resolution image
        img = Input(shape=self.shape_hr)
        x = conv3d_block(img, filters, bn=False)
        x = conv3d_block(x, filters, strides=2)
        x = conv3d_block(x, filters * 2)
        x = conv3d_block(x, filters * 2, strides=2)
        x = conv3d_block(x, filters * 4)
        x = conv3d_block(x, filters * 4, strides=2)
        x = conv3d_block(x, filters * 8)
        x = conv3d_block(x, filters * 8, strides=2)
        x = Dense(filters * 16)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.4)(x)
        x = Dense(1)(x)

        # Create model and compile
        model = Model(inputs=img, outputs=x)
        #model.summary()
        return model

    def build_piesrgan(self):
        """Create the combined PIESRGAN network"""
        def comput_loss(x):
            img_hr, generated_hr = x 
            # Compute the Perceptual loss ###based on GRADIENT-field MSE
            grad_hr_1 = tf.compat.v1.py_func(grad1,[img_hr], tf.float32)
            grad_hr_2 = tf.compat.v1.py_func(grad2,[img_hr],tf.float32)
            grad_hr_3 = tf.compat.v1.py_func(grad3,[img_hr],tf.float32)
            grad_sr_1 = tf.compat.v1.py_func(grad1,[generated_hr], tf.float32)
            grad_sr_2 = tf.compat.v1.py_func(grad2,[generated_hr],tf.float32)
            grad_sr_3 = tf.compat.v1.py_func(grad3,[generated_hr],tf.float32)
            #grad_loss = K.mean(tf.losses.mean_squared_error( generated_hr, img_hr))
            #grad_loss= tf.py_function(grad1,[tf.math.subtract(img_hr,generated_hr)],tf.float32)
            #grad_loss=tf.math.reduce_mean(grad)
            grad_loss = K.mean( 
                tf.losses.mean_squared_error(grad_hr_1,grad_sr_1)+
                tf.losses.mean_squared_error(grad_hr_2,grad_sr_2)+
                tf.losses.mean_squared_error(grad_hr_3,grad_sr_3))
            # Compute the RaGAN loss
            fake_logit, real_logit = self.RaGAN([img_hr, generated_hr])
            gen_loss = K.mean(
                K.binary_crossentropy(K.zeros_like(real_logit), real_logit) +
                K.binary_crossentropy(K.ones_like(fake_logit), fake_logit))
            # Compute the pixel_loss with L1 loss
            pixel_loss = K.mean(tf.losses.mean_squared_error(generated_hr, img_hr))
            return [grad_loss, gen_loss, pixel_loss]
        
        # Input LR images
        img_lr = Input(shape=self.shape_lr)
        img_hr = Input(shape=self.shape_hr)
        # Create a high resolution image from the low resolution one
        generated_hr = self.generator(img_lr)
        # In the combined model we only train the generator
        self.discriminator.trainable = False
        self.RaGAN.trainable = False
              
        # Output tensors to a Model must be the output of a Keras `Layer`
        total_loss = Lambda(comput_loss, name='comput_loss')([img_hr, generated_hr])
        grad_loss = Lambda(lambda x: self.loss_weights['percept'] * x, name='grad_loss')(total_loss[0])
        gen_loss = Lambda(lambda x: self.loss_weights['gen'] * x, name='gen_loss')(total_loss[1])
        pixel_loss = Lambda(lambda x: self.loss_weights['pixel'] * x, name='pixel_loss')(total_loss[2])
        outputs = Lambda(lambda x : x, name = 'output')(total_loss)
        #loss = Lambda(lambda x: self.loss_weights['percept']*x[0]+self.loss_weights['gen']*x[1]+self.loss_weights['pixel']*x[2], name='total_loss')(total_loss)
       
        # Create model
        model = Model(inputs=[img_lr, img_hr], outputs=[grad_loss, gen_loss, pixel_loss])

        # Add the loss of model and compile
        #model.add_loss(loss)
        model.add_loss(grad_loss)
        model.add_loss(gen_loss)
        model.add_loss(pixel_loss)
        #model.metrics_tensors = []
        # Create metrics of PIESRGAN
        model.add_metric(grad_loss, name='grad_loss')
        model.add_metric(gen_loss, name='gen_loss')
        model.add_metric(gen_loss, name='pixel_loss')

        model.metrics_names.append('grad_loss')
        model.metrics_names.append('gen_loss')
        model.metrics_names.append('pixel_loss')
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        model.compile(optimizer=Adam(self.gen_lr))

        #model.metrics_tensors.append(pixel_loss) 
        #model.summary()
        return model
        
    def build_RaGAN(self):
        def interpolating(x):
            u = K.random_uniform((K.shape(x[0])[0],) + (1,) * (K.ndim(x[0]) - 1))
            return x[0] * u + x[1] * (1 - u)

        def comput_loss(x):
            real, fake = x
            fake_logit = (fake - K.mean(real))
            real_logit = (real - K.mean(fake))
            # This should return sigmoid of the values
            return [tf.math.sigmoid(fake_logit), tf.math.sigmoid(real_logit)]

        # Input HR images
        imgs_hr = Input(self.shape_hr)
        generated_hr = Input(self.shape_hr)
        
        # Create a high resolution image from the low resolution one
        real_discriminator_logits = self.discriminator(imgs_hr)
        fake_discriminator_logits = self.discriminator(generated_hr)

        total_loss = Lambda(comput_loss, name='comput_loss')([real_discriminator_logits, fake_discriminator_logits])
        # Output tensors to a Model must be the output of a Keras `Layer`
        fake_logit = Lambda(lambda x: x, name='fake_logit')(total_loss[0])
        real_logit = Lambda(lambda x: x, name='real_logit')(total_loss[1])
        dis_loss = K.mean(K.binary_crossentropy(K.zeros_like(fake_logit), fake_logit) +
                          K.binary_crossentropy(K.ones_like(real_logit), real_logit))
        # dis_loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logit), logits=fake_logit) +
        #     tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_likes(real_logit), logits=real_logit))
        # dis_loss = K.mean(- (real_logit - fake_logit)) + 10 * K.mean((grad_norms - 1) ** 2)

        model = Model(inputs=[imgs_hr, generated_hr], outputs=[fake_logit, real_logit])

        model.add_loss(dis_loss)
        model.add_metric(dis_loss,name='diss_loss')
        model.compile(optimizer=Adam(self.dis_lr))

        model.metrics_names.append('dis_loss')
        model.metrics_tensors = []
        model.metrics_tensors.append(dis_loss)
        return model

  
    def compile_generator(self, model):
        """Compile the generator with appropriate optimizer"""
        def pixel_loss(y_true, y_pred):
             loss1=tf.losses.mean_squared_error(y_true,y_pred)
             loss2=tf.compat.v1.losses.absolute_difference(y_true,y_pred)
             return loss1
             #return 1*loss1+0.005*loss2
        def mae_loss(y_true, y_pred):
            loss=tf.losses.absolute_difference(y_true,y_pred)
            return loss*0.01
        def grad_loss(y_true,y_pred):
            grad_hr_1 = tf.py_function(grad1,[y_true],tf.float32)
            grad_hr_2 = tf.py_function(grad2,[y_true],tf.float32)
            grad_hr_3 = tf.py_function(grad3,[y_true],tf.float32)
            grad_sr_1 = tf.py_function(grad1,[y_pred],tf.float32)
            grad_sr_2 = tf.py_function(grad2,[y_pred],tf.float32)
            grad_sr_3 = tf.py_function(grad3,[y_pred],tf.float32)
            grad_loss = K.mean( 
                tf.losses.mean_squared_error(grad_hr_1,grad_sr_1)+
                tf.losses.mean_squared_error(grad_hr_2,grad_sr_2)+
                tf.losses.mean_squared_error(grad_hr_3,grad_sr_3))
            return grad_loss
       
        model.compile(
            loss=pixel_loss,
            optimizer=Adam(self.gen_lr, 0.9,0.999),
            metrics=['mse']
            #metrics=['mse','mae', self.PSNR]
        )

    def compile_discriminator(self, model):
        """Compile the generator with appropriate optimizer"""
        model.compile(
            loss=None,
            optimizer=Adam(self.dis_lr, 0.9, 0.999),
            metrics=['accuracy']
        )

    def compile_piesrgan(self, model):
        """Compile the PIESRGAN with appropriate optimizer"""
        model.compile(
            loss=None,
            optimizer=Adam(self.gen_lr, 0.9, 0.999)
        )

    def PSNR(self, y_true, y_pred):
        """
        Peek Signal to Noise Ratio
        PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)

        Since input is scaled from -1 to 1, MAX_I = 1, and thus 20 * log10(1) = 0. Only the last part of the equation is therefore neccesary.
        """
        return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)
        
    def train_generator(self,
        epochs, batch_size,boxsize,
        workers=1,
        use_multiprocessing=False,
        dataname='doctor',
        datapath_train=None,
        datapath_validation='../../../hpcwork/zl963564/box5500.h5',
        datapath_test='../../../hpcwork/zl963564/box5500.h5',
        steps_per_epoch=1,
        steps_per_validation=4,
        crops_per_image=2,
        log_weight_path='./data/weights/',
        log_tensorboard_path='./data/logs/',
        log_tensorboard_name='SR-RRDB-D',
        log_tensorboard_update_freq=20,
        log_test_path="./images/samples-d/",
        nxg=1536, nyg=1536, nzg=1536
        ):
        """Trains the generator part of the network with MSE loss"""

        # Create data loaders
        #f=h5.File(datapath_train,'r')
        # One for hr one for lr
        train_loader = [DataLoader_s3d(datapath_train+'/DNS/s-1.5000E-05', nxg, nyg, nzg, 2, batch_size, boxsize), \
                        DataLoader_s3d(datapath_train+'/Filt_8x/filt_s-1.5000E-05', nxg, nyg, nzg, 2, batch_size, boxsize)]
        test_loader  = [DataLoader_s3d(datapath_test+'/DNS/s-2.4500E-05', nxg, nyg, nzg, 2, batch_size, boxsize), \
                        DataLoader_s3d(datapath_test+'/Filt_8x/filt_s-2.4500E-05', nxg, nyg, nzg, 2, batch_size, boxsize)]

        mins, maxs = train_loader[0].get_norm_constants()
        mins = mins[0]
        maxs = maxs[0]
        for step in range(epochs // 10):
            with tf.device('/cpu:0'):
                epoch_starttime=datetime.datetime.now()
            self.compile_generator(self.generator)             
            # Callback: tensorboard
            callbacks = []
            if log_tensorboard_path:
                tensorboard = TensorBoard(
                    log_dir=os.path.join(log_tensorboard_path, log_tensorboard_name),
                    histogram_freq=0,
                    batch_size=batch_size,
                    write_graph=False,
                    write_grads=False,
                    update_freq=log_tensorboard_update_freq
                )
                callbacks.append(tensorboard)
            else:
                print(">> Not logging to tensorboard since no log_tensorboard_path is set")

            # Callback: save weights after each epoch
            modelcheckpoint = ModelCheckpoint(
                os.path.join(log_weight_path, dataname + '_{}X.h5'.format(self.upscaling_factor)),
                monitor='PSNR',
                save_best_only=True,
                save_weights_only=True
            )
            callbacks.append(modelcheckpoint)
            csv_logger = CSVLogger("model_history_log.csv", append=True)

            # Fit the model
            for i in range(5,6):
                ''' ##  iterate from lmbda= 64 to 64 '''

                filter_nr=2**i
                sum_load=datetime.timedelta(seconds=0,microseconds=0)
                sum_train=datetime.timedelta(seconds=0,microseconds=0)
                with tf.device('/cpu:0'):
                    print(">> fitting lmbda = ",filter_nr)
                    print(">> using [batch size = ",batch_size,"] and  [sub-boxsize = 16]")
                #for idx in range (int(64*64*0/batch_size),int(64*64*31/batch_size)):
                for idx in range(train_loader[0].nbatches):
                #for idx in range(10):
                    if idx%20==0:
                        self.generator.save_weights("{}_generator_idx{}.h5".format(log_weight_path,idx))
                    with tf.device('/cpu:0'):
                        batch_starttime = datetime.datetime.now()
                        hr_train = do_normalisation(train_loader[0].getData(0), 'minmax', mins, maxs)
                        hr_test  = do_normalisation(test_loader[0].getData(0), 'minmax', mins, maxs)
                        lr_train = do_normalisation(train_loader[1].getData(0), 'minmax', mins, maxs)
                        lr_test  = do_normalisation(test_loader[1].getData(0), 'minmax', mins, maxs)
                        loading_time=datetime.datetime.now()
                        test_data = lr_test, hr_test
                    with tf.device('/cpu:0'):
                        temp1=loading_time-batch_starttime
                        sum_load=sum_load+temp1
                        print(">>---------->>>>>>>fitting on batch #",idx,"/",(train_loader[0].nbatches)," batch loaded in ",temp1,"s")
                        #print(">>---------->>>>>>>>>>>>>>>[ts=8000]" )
                    self.generator.fit(
                    lr_train, hr_train,
                    steps_per_epoch=steps_per_epoch,
                    epochs=3,
                    #validation_data=test_data,
                    #validation_steps=steps_per_validation,
                    #callbacks=[csv_logger, tensorboard],
                    callbacks = [csv_logger],
                    #callbacks = callbacks,
                    use_multiprocessing=use_multiprocessing,
                    workers=workers
                    )
                    with tf.device('/cpu:0'):
                        fitted_time=datetime.datetime.now()
                        temp2=fitted_time-loading_time
                        sum_train=sum_train+temp2
                        print(">>---------->>>>>>>batch #",idx,"/",(train_loader[0].nbatches)," fitted in ",temp2,"s")
                        if idx%5==0:
                            print("[Summary] at idx=",idx," ,[total loading time]=",sum_load, " [total training time]=",sum_train)
                            gc.collect()
                print(">> lambda = ",filter_nr, " trained")
                self.generator.save('./data/weights/DNS_generator_lmbda64.h5')
            #self.gen_lr /= 1.149
            print(step, self.gen_lr)   
            
            
            
    def train_piesrgan(self,
        epochs, batch_size,
        dataname,
        datapath_train='../in3000.h5',
        datapath_validation='../in3000.h5',
        steps_per_validation=10,
        datapath_test='../in3000.h5',
        workers=1, max_queue_size=1,
        first_epoch=0,
        print_frequency=2,
        crops_per_image=2,
        log_weight_frequency=2,
        log_weight_path='./data/weights/',
        log_tensorboard_path='./data/logs_1/',
        log_tensorboard_name='PIESRGAN',
        log_tensorboard_update_freq=4,
        log_test_frequency=4,
        log_test_path="./images/samples/",
        nxg=1536, nyg=1536, nzg=1536
        ):
        """Train the PIESRGAN network
        :param int epochs: how many epochs to train the network for
        :param str dataname: name to use for storing model weights etc.
        :param str datapath_train: path for the image files to use for training
        :param str datapath_test: path for the image files to use for testing / plotting
        :param int print_frequency: how often (in epochs) to print progress to terminal. Warning: will run validation inference!
        :param int log_weight_frequency: how often (in epochs) should network weights be saved. None for never
        :param int log_weight_path: where should network weights be saved
        :param int log_test_frequency: how often (in epochs) should testing & validation be performed
        :param str log_test_path: where should test results be saved
        :param str log_tensorboard_path: where should tensorflow logs be sent
        :param str log_tensorboard_name: what folder should tf logs be saved under
        """
        # Each epoch == "update iteration" as defined in the paper
        print_losses = {"G": [], "D": []}
        start_epoch = datetime.datetime.now()

        # Random images to go through
        #idxs = np.random.randint(0, len(loader), epochs)
        # One for hr one for lr
        train_loader = [DataLoader_s3d(datapath_train+'/DNS/s-1.5000E-05', nxg, nyg, nzg, 2, batch_size, 16), \
                        DataLoader_s3d(datapath_train+'/Filt_8x/filt_s-1.5000E-05', nxg, nyg, nzg, 2, batch_size, 16)]
        test_loader  = [DataLoader_s3d(datapath_test+'/DNS/s-2.4500E-05', nxg, nyg, nzg, 2, batch_size, 16), \
                        DataLoader_s3d(datapath_test+'/Filt_8x/filt_s-2.4500E-05', nxg, nyg, nzg, 2, batch_size, 16)]
        mins, maxs = train_loader[0].get_norm_constants()
        mins = mins[0]
        maxs = maxs[0]
        logging.basicConfig(filename='GAN_logs')
        # Loop through epochs / iterations
        for epoch in range(first_epoch, epochs + first_epoch):
            # Start epoch time
            if epoch % print_frequency == 0:
                start_epoch = datetime.datetime.now()
            for lmd in range(5,6):
                filter_nr=2**lmd
                print(">> fitting lmbda = ",filter_nr)
                #imgs_lr,imgs_hr = RandomLoader_train(datapath_train, '{0:03}'.format(filter_nr),batch_size)
                imgs_lr = do_normalisation(train_loader[1].getData(0),'minmax', mins, maxs)
                imgs_hr = do_normalisation(train_loader[0].getData(0),'minmax',mins,maxs)
                generated_hr = self.generator.predict(imgs_lr,steps=1)
                
                for step in range(10):
                # SRGAN's loss (don't use them)
                # real_loss = self.discriminator.train_on_batch(imgs_hr, real)
                # fake_loss = self.discriminator.train_on_batch(generated_hr, fake)
                # discriminator_loss = 0.5 * np.add(real_loss, fake_loss)
                    #print("step: ",step+1)
                # Train Relativistic Discriminator
                    discriminator_loss = self.RaGAN.train_on_batch([imgs_hr, generated_hr], None)
            
                # Train generator
                # features_hr = self.vgg.predict(self.preprocess_vgg(imgs_hr))
                    generator_loss = self.piesrgan.train_on_batch([imgs_lr, imgs_hr], None)
                # Save losses
                    print_losses['G'].append(generator_loss)
                    print_losses['D'].append(discriminator_loss)
                    #print(discriminator_loss)
                # Show the progress
            if epoch % print_frequency == 0:
                g_avg_loss = np.array(print_losses['G']).mean(axis=0)
                d_avg_loss = np.array(print_losses['D']).mean(axis=0)
                #print(self.piesrgan.metrics_names)
                #print(g_avg_loss)
                #print(self.piesrgan.metrics_names, g_avg_loss)
                #print(self.RaGAN.metrics_names, d_avg_loss)
                print("\nEpoch {}/{} | Time: {}s\n>> Generator/GAN: {}\n>> Discriminator: {}".format(
                   epoch, epochs + first_epoch,
                    (datetime.datetime.now() - start_epoch).seconds,
                    ", ".join(["{}={:.5f}".format(k, v) for k, v in zip(self.piesrgan.metrics_names, g_avg_loss)]),
                    ", ".join(["{}={:.5f}".format(k, v) for k, v in zip(self.RaGAN.metrics_names, d_avg_loss)])
                ))
                print(g_avg_loss[0], "LOSS")
                logging.warning("%s,%s", g_avg_loss[0], d_avg_loss[0])
                print_losses = {"G": [], "D": []}
            # Check if we should save the network weights
            if log_weight_frequency and epoch % log_weight_frequency == 0:
                # Save the network weights
                print(">> Saving the network weights")
                self.save_weights(os.path.join(log_weight_path, dataname), epoch)         
                
    def test(self,
        refer_model=None,
        batch_size=1,
        datapath_test='../../../../scratch/cjhpc55/jhpc5502/filt/R1/output_0000005500.h5',
        boxsize=128,
        output_name=None
        ):
        """Trains the generator part of the network"""
        f=h5.File(datapath_test,'r')
        # Create data loaders
        
        #High-resolution box
        box_hr = f['ps/ps_01'][boxsize*5:boxsize*(5+1), boxsize*5:boxsize*(5+1), boxsize*1:boxsize*(1+1)] 
        #Low-resolution box
        box_lr = f['kc_000064/ps'][boxsize*5:boxsize*(5+1), boxsize*5:boxsize*(5+1), boxsize*1:boxsize*(1+1)] 
        lr_input = tf.Variable(tf.zeros([1,boxsize,boxsize,boxsize,1] )) 
        K.set_value(lr_input[0,0:boxsize,0:boxsize,0:boxsize,0],box_lr)
        #Reconstruction
        sr_output = self.generator.predict(lr_input,steps=1)
        box_sr = sr_output[0,:,:,:,0]
        #create image slice for visualization
        img_hr = box_hr[:,:,64]
        img_lr = box_lr[:,:,64]
        img_sr = box_sr[:,:,64]

        print(">> Ploting test images")
        Image_generator(img_lr,img_sr,img_hr,output_name)        




#here starts python execution commands
# Run the PIESRGAN network
if __name__ == '__main__':
   
    t1 = time.time()
    print(">> Creating the PIESRGAN network")
    gan = PIESRGAN(training_mode=True,
                gen_lr=1e-4, dis_lr=1e-4,
                channels=3
                )
    # # Stage1: Train the generator w.r.t RRDB first
    print(">> Start training generator")
    print(">> training [ts=5500]")
    print("GAN creation took {} secs".format(time.time()-t1))
    #gan.train_generator(
    #     epochs=10,
    #     datapath_train='../../../hpcwork/zl963564/box5500.h5',
    #     batch_size=32,
    #)
# Deepak
    gan.train_generator(
         epochs=10,
         boxsize=32,
         datapath_train='/scratch/w47/share/IsotropicTurb',
         datapath_test='/scratch/w47/share/IsotropicTurb',
         batch_size=32,
         steps_per_validation=1,
         steps_per_epoch=1,
         workers=1,
         log_weight_path='./data/weights2/',
         use_multiprocessing=True
    )

    gan.load_weights(generator_weights='./data/weights/_generator_idx3320.h5')
    '''    
    t1=time.time()
    gan.load_weights(generator_weights='./data/weights/_generator_idx900.h5')
    print("Loading weights took {} secs".format(time.time()-t1))
    train_loader = DataLoader_s3d('/scratch/w47/share/IsotropicTurb/Filt_8x/filt_s-1.5000E-05', 1536, 1536, 1536, 2, 144, 16)
    train_loader_hr = DataLoader_s3d('/scratch/w47/share/IsotropicTurb/DNS/s-1.5000E-05', 1536, 1536, 1536, 2, 144, 16)
    mins, maxs = train_loader_hr.get_norm_constants()
    mins = mins[0]
    maxs = maxs[0]

    #data = do_normalisation(train_loader.getData(0), 'minmax', mins, maxs)
    #data_hr = do_normalisation(train_loader_hr.getData(0), 'minmax', mins, maxs)
    data = np.zeros([1536, 1536, 16,3],dtype=np.float32)
    for i in range(16):
        data[:,:,i,:] = train_loader.get_data_plane(i,3)

    data = train_loader.reshape_array([16, 16, 16],data)
    #print(data.shape) 
    #data = do_normalisation(train_loader.get_data_plane(0,3), 'minmax', mins, maxs)
    #data_hr = do_normalisation(train_loader_hr.get_data_plane(0,3), 'minmax', mins, maxs)
    #pred = gan.generator.predict(data[None,0:1000,0:1000,None,:])
    #print(data.shape)
    t1=time.time()
    pred = gan.generator.predict(data[:,:,:,:,:], workers=4, use_multiprocessing=True)
    print("Time to predict {}".format(time.time()-t1))
    print(pred.shape)
    '''
    #Image_generator(data[0:1000,0:1000,0],pred[0,0:1000,0:1000,0,0],data_hr[0:1000,0:1000,0],'testfig.pdf')        
    #Image_generator(data[0,:,:,0,0],pred[0,:,:,0,0],data_hr[0,:,:,0,0],'testfig.pdf')        
    #print(">> Generator trained based on MSE")
    '''Save pretrained generator'''
    #gan.generator.save('./data/weights/Doctor_gan.h5')
    #gan.save_weights('./data/weights/')
    # Stage2: Train the PIESRGAN with percept_loss, gen_loss and pixel_loss
    #print(">> Start training PIESRGAN")
    #gan.generator.summary()    
    #gan.train_piesrgan(
    #     epochs=500,
    #     first_epoch=0,
    #     batch_size=8,
    #     dataname='IsoTurb',
        #datapath_train='../datasets/DIV2K_224/',
    #     datapath_train='/scratch/w47/share/IsotropicTurb',
    #     datapath_validation='/scratch/w47/share/IsotropicTurb',
    #     datapath_test='/scratch/w47/share/IsotropicTurb',
    ##     print_frequency=2
    #     )
    #print(">> Done with PIESRGAN training")
    #gan.save_weights('./data/weights/')


    # Stage 3: Testing
    #print(">> Start testing PIESRGAN")
    #gan.test(output_name='test_1.png')
    #print(">> Test finished, img file saved at: <test_1.png>")

