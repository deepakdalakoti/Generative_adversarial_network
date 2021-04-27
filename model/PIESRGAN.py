#! /usr/bin/python
# -*- coding: utf-8 -*-
# ! /usr/bin/python
import os          # enables interactions with the operating system
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
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
from keras.layers import BatchNormalization, LeakyReLU, PReLU, Conv2D, Dense, Conv3D,ZeroPadding3D, Flatten
from keras.layers import UpSampling2D, Lambda, Dropout, Conv3DTranspose
from keras.optimizers import Adam, RMSprop
from keras.applications.vgg19 import preprocess_input
from keras.utils.data_utils import OrderedEnqueuer
from tensorflow.keras import backend as K           #campatability
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
from keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import CSVLogger
sys.stderr = stderr
from util import UpSampling3D, DataLoader, RandomLoader_train, Image_generator, subPixelConv3d, subPixelConv3d2
#subPixelConv3d, DataLoader, RandomLoader_train, Image_generator
from util import DataLoader_s3d, do_normalisation, pixelShuffler
import h5py as h5
import numpy as np
'''Use Horovod in case of multi nodes parallelizations'''
#import horovod.keras as hvd
#import horovod.tensorflow.keras as hvd

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
                 upscaling_factor=8,
                 # VGG scaled with 1/12.75 as in paper
                 loss_weights={'percept':5e-1,'gen':5e-5, 'pixel':1e-2},
                 training_mode=True,
                 refer_model=None,
                 RRDB_layers=3
                 ):
        """
        :param int height_lr: Height of low-resolution DNS data
        :param int width_lr: Width of low-resolution DNS data
        :param int depth: Width of low-resolution DNS data
        :param int upscaling_factor: Up-scaling factor
        :param int gen_lr: Learning rate of generator
        :param int dis_lr: Learning rate of discriminator
        """
        #hvd.init()

        #gpus = tf.config.experimental.list_physical_devices('GPU')
        #for gpu in gpus:
        #    tf.config.experimental.set_memory_growth(gpu, True)
        #if gpus:
        #    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

        self.mirrored_strategy = tf.distribute.MirroredStrategy()
        self.upscaling_factor=upscaling_factor
        # Low-resolution image dimensions
        self.height_lr = height_lr
        self.width_lr = width_lr
        self.depth_lr=depth_lr
        self.channels_lr = int(channels)
        self.RRDB_layers = RRDB_layers
        # High-resolution image dimensions are upscaling_factor times the LR ones
        self.height_hr = int(self.height_lr*self.upscaling_factor )
        self.width_hr = int(self.width_lr*self.upscaling_factor )
        self.depth_hr = int(self.depth_lr*self.upscaling_factor )
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
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.gen_lr,beta_1=0.9,beta_2=0.999)
            #self.compile_generator(self.generator)
            #self.refer_model = refer_model
            # If training, build rest of GAN network
        #    if training_mode:
        #        self.discriminator = self.build_discriminator()
        #        self.compile_discriminator(self.discriminator)
        #        self.RaGAN = self.build_RaGAN()
        #        self.piesrgan = self.build_piesrgan()
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
        #w_init = tf.random_normal_initializer(stddev=0.02)
        w_init = tf.keras.initializers.HeNormal()
        #w_init = tf.keras.initializers.GlorotUniform()
        height_hr=self.height_hr
        width_hr=self.width_hr
        depth_hr=self.depth_hr
        beta = 0.2
        slope= 0.1
        def dense_block(input):
            x1 = Conv3D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=w_init)(input)
            #x1 = BatchNormalization()(x1)
            x1 = LeakyReLU(slope)(x1)
            x1 = Concatenate()([input, x1])

            x2 = Conv3D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=w_init)(x1)
            #x2 = BatchNormalization()(x2)
            x2 = LeakyReLU(slope)(x2)
            x2 = Concatenate()([input, x1, x2])

            x3 = Conv3D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=w_init)(x2)
            #x3 = BatchNormalization()(x3)
            x3 = LeakyReLU(slope)(x3)
            x3 = Concatenate()([input, x1, x2, x3])

            x4 = Conv3D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=w_init)(x3)
            #x4 = BatchNormalization()(x4)
            x4 = LeakyReLU(slope)(x4)
            x4 = Concatenate()([input, x1, x2, x3, x4])  #added x3, which ESRGAN didn't include

            x5 = Conv3D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=w_init)(x4)
            #x5 = BatchNormalization()(x5)
            x5 = Lambda(lambda x: x * beta)(x5)
            """here: assumed beta=0.2"""
            x = Add()([x5, input])
            return x

        def RRDB(input, layers=12):
            # How many layers?
            x = dense_block(input)
            for i in range(layers-1):
                x = dense_block(x)
            #x = dense_block(x)
            """here: assumed beta=0.2 as well"""
            x = Lambda(lambda x: x * beta)(x)
            out = Add()([x, input])
            return out

        def residual_block(x_in):
            x = Conv3D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=w_init)(x_in)
            #x = BatchNormalization()(x)
            x = LeakyReLU(slope)(x)

            x = Conv3D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=w_init)(x)
            #x = BatchNormalization()(x)
            x = Add()([x, x_in])

            return x


        """----------------Assembly the generator-----------------"""
        # Input low resolution image
        #lr_input = Input(shape=(None, None, None,3))
        #with self.mirrored_strategy.scope():
        lr_input = Input(shape=self.shape_lr)
        # Pre-residual
        #x_start = Conv3D(64, data_format="channels_last", kernel_size=3, strides=1, padding='same')(lr_input)
        # kernel size 9 in original paper here
        x_start = Conv3D(64, data_format="channels_last", kernel_size=3, strides=1, padding='same', kernel_initializer=w_init)(lr_input)
        x_start = LeakyReLU(slope)(x_start)

        # Residual-in-Residual Dense Block
        x = RRDB(x_start, self.RRDB_layers)
        #x = residual_block(x_start)
        #for i in range(self.RRDB_layers):
        #    x = residual_block(x)

        # Post-residual block
        x = Conv3D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=w_init)(x)
        #x = BatchNormalization()(x)
        x = Lambda(lambda x: x * beta)(x)
        x = Add()([x, x_start])
        #x = Conv3D(512,kernel_size=3, strides=1, padding='same',activation=lrelu1)(x)
        #x = Conv3D(512,kernel_size=3, strides=1, padding='same',activation=lrelu1)(x)
        # Be consistent with the original paper
        # Upsample now, for 8x
        x = Conv3D(256,kernel_size=3, strides=1, padding='same', kernel_initializer=w_init)(x)
        #x = subPixelConv3d(x, self.height_hr, self.width_hr, self.depth_hr, 3, 32)
        #x = subPixelConv3d2(x, 32)
        x = pixelShuffler(x, scale=2)
        #x = Conv3DTranspose(filters = 32, kernel_size=6, strides=2, padding='same')(x)
        x = LeakyReLU(slope)(x)

        x = Conv3D(256,kernel_size=3, strides=1, padding='same', kernel_initializer=w_init)(x)
        #x = subPixelConv3d(x, self.height_hr, self.width_hr, self.depth_hr, 2, 32)
        #x = subPixelConv3d2(x, 32)
        #x = Conv3DTranspose(filters = 32, kernel_size=6, strides=2, padding='same')(x)
        x = pixelShuffler(x)
        x = LeakyReLU(slope)(x)

        #x = Conv3D(16,kernel_size=3, strides=1, padding='same', kernel_initializer=w_init)(x)
        #x = Conv3D(8,kernel_size=3, strides=1, padding='same', kernel_initializer=w_init)(x)
        #x = Conv3D(4,kernel_size=3, strides=1, padding='same', kernel_initializer=w_init)(x)
        #x = Conv3D(2,kernel_size=3, strides=1, padding='same', kernel_initializer=w_init)(x)
        #Final 2 convolutional layers
        #x = Conv3D(16, kernel_size=3, strides=1, padding='same')(x)
        #x = pixelShuffler(x)
        #x = subPixelConv3d(x, self.height_hr, self.width_hr, self.depth_hr, 1, 32)
        #x = subPixelConv3d(x,  32)
        #x = UpSampling3D()(x)
        #x = LeakyReLU(slope)(x)

        #x = Conv3D(8, kernel_size=3, strides=1, padding='same')(x)
        #x = LeakyReLU(slope)(x)
        # Activation for output?
        #hr_output = Conv3D(self.channels_hr, kernel_size=3, strides=1, padding='same', activation='tanh')(x)
        #hr_output = Conv3D(self.channels_hr, kernel_size=3, strides=1, padding='same')(x)
        # 9 kernel size for last layer in original paper
        hr_output = Conv3D(self.channels_hr, kernel_size=3, strides=1, padding='same', kernel_initializer=w_init)(x)

        # Create model and compile
        model = Model(inputs=lr_input, outputs=hr_output)

        #wghts = np.array(model.get_weights())
        #model.set_weights(wghts*0.1)
        #Uncomment this if using multi GPU model
        #model=multi_gpu_model(model,gpus=2,cpu_merge=True)
        #model.summary()
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
        #x = conv3d_block(x, filters * 8, strides=2)
        x = Flatten()(x)
        x = Dense(filters * 16)(x)
        x = LeakyReLU(alpha=0.2)(x)
        # No dropout in original model
        #x = Dropout(0.4)(x)
        x = Dense(1)(x)

        # Create model and compile
        model = Model(inputs=img, outputs=x)
        model.summary()
        return model
    
    def build_gan(self, gen_weights=None, disc_weights=None):

        with self.mirrored_strategy.scope():
            self.gen_gan = self.build_generator()
            self.dis_gan = self.build_discriminator()
            if(gen_weights):
                self.gen_gan.load_weights(gen_weights)
            if(disc_weights):
                self.dis_gan.load_weights(disc_weights)

            self.generator_optimizer = tf.keras.optimizers.Adam(self.gen_lr, beta_1=0.9, beta_2=0.999)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(self.dis_lr, beta_1=0.9, beta_2=0.999)

    def train_gan_step(self, img_lr, img_hr):
        #with self.mirrored_strategy.scope():
        def grad_all(ytrue, ypred):
            loss = (np.gradient(ytrue,axis=1)-np.gradient(ypred,axis=1))**2 + \
                   (np.gradient(ytrue,axis=2)-np.gradient(ypred,axis=2))**2 + \
                   (np.gradient(ytrue,axis=3)-np.gradient(ypred,axis=3))**2 
            loss = np.mean(loss, axis=(1,2,3,4))
            #loss = np.mean(loss, axis=-1)
            return loss
        def continuity_loss(ytrue, ypred):
            # Continuity loss
            #mass_true = np.gradient(ytrue[:,:,:,:,0],axis=1) + np.gradient(ytrue[:,:,:,:,1],axis=2) + \
            #            np.gradient(ytrue[:,:,:,:,2],axis=3)
            #mass_pred = np.gradient(ypred[:,:,:,:,0],axis=1) + np.gradient(ypred[:,:,:,:,1],axis=2) + \
            #            np.gradient(ypred[:,:,:,:,2],axis=3)
            #loss = np.mean((mass_true-mass_pred)**2, axis=(1,2,3))
            loss = np.mean(np.abs(ytrue-ypred), axis=(1,2,3,4))
            return loss

        def comput_loss(x):
            # Need per sample loss, i.e. one value for one sample
            img_hr, generated_hr, fake, real = x 
            fake_logit = tf.math.sigmoid(fake-K.mean(real))
            real_logit = tf.math.sigmoid(real - K.mean(fake))
            grad_loss = tf.nn.compute_average_loss(tf.compat.v1.py_func(grad_all,[img_hr, generated_hr], tf.float32))
            cont_loss = tf.nn.compute_average_loss(tf.compat.v1.py_func(continuity_loss,[img_hr, generated_hr], tf.float32))
            BCE = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
            gen_loss =  tf.nn.compute_average_loss(BCE(tf.zeros_like(real_logit), real_logit) +
                    BCE(tf.ones_like(fake_logit), fake_logit))
            #mse = tf.keras.losses.MeanSquaredError(tf.keras.losses.Reduction.NONE)
            mse_loss = tf.math.reduce_mean(tf.math.square(img_hr - generated_hr), axis=(1,2,3,4))
            #pixel_loss = tf.nn.compute_average_loss(mse(img_hr,generated_hr))
            pixel_loss = tf.nn.compute_average_loss(mse_loss)
            #print(gen_loss, grad_loss)
            total = grad_loss + gen_loss*0.005 + pixel_loss + cont_loss*0.0
            return total, grad_loss, gen_loss, pixel_loss, cont_loss
            #return grad_loss+gen_loss+pixel_loss
            #return [grad_loss, gen_loss, pixel_loss]
        def comput_loss_disc(x):
            real, fake = x
            fake_logit = tf.math.sigmoid(fake - K.mean(real))
            real_logit = tf.math.sigmoid(real - K.mean(fake))
            BCE = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
            dis_loss = tf.nn.compute_average_loss(BCE(K.zeros_like(fake_logit), fake_logit) +
                          BCE(K.ones_like(real_logit), real_logit))
            return dis_loss, fake_logit, real_logit

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape: 
            generated_hr = self.gen_gan(img_lr, training=True) 
            real_discriminator_logits = self.dis_gan(img_hr, training=True)
            fake_discriminator_logits = self.dis_gan(generated_hr, training=True)
            disc_loss, fake_logits, real_logits = comput_loss_disc([real_discriminator_logits, fake_discriminator_logits])
            total_loss, grad_loss, gen_loss, pixel_loss, cont_loss = comput_loss([img_hr, generated_hr, fake_discriminator_logits, real_discriminator_logits ])

        gradients_of_generator = gen_tape.gradient(total_loss, self.gen_gan.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.dis_gan.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.gen_gan.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.dis_gan.trainable_variables))

        return total_loss, grad_loss, gen_loss, pixel_loss, cont_loss, disc_loss

    @tf.function
    def distributed_train_step(self,imgs_lr,  imgs_hr):
        total_loss_per, grad_loss_per, gen_loss_per, pixel_loss_per, \
                cont_loss_per, disc_loss_per = self.mirrored_strategy.run(self.train_gan_step, args=(imgs_lr, imgs_hr,))
        total_loss = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, total_loss_per,axis=None)
        grad_loss = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, grad_loss_per,axis=None)
        gen_loss = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, gen_loss_per,axis=None)
        pixel_loss = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, pixel_loss_per,axis=None)
        cont_loss = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, cont_loss_per,axis=None)
        disc_loss = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, disc_loss_per,axis=None)

        return total_loss, grad_loss, gen_loss, pixel_loss, cont_loss, disc_loss

    def train_gen_step(self, img_lr, img_hr):
        def grad_loss_gen(y_true,y_pred):
            #mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            #total_loss = mse(y_true,y_pred)
            total_loss = tf.math.reduce_mean(tf.math.square(y_true-y_pred), axis=(1,2,3,4))
            return tf.nn.compute_average_loss(total_loss, global_batch_size=self.batch_size)

        with tf.GradientTape() as gen_tape:
            generated_hr = self.generator(img_lr, training=True) 
            loss = grad_loss_gen(img_hr, generated_hr)
        
        gradients_of_generator = gen_tape.gradient(loss, self.generator.trainable_variables)
        #print(loss)
        self.optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return loss
    
    @tf.function
    def distributed_train_gen(self, imgs_lr,  imgs_hr):
        total_loss_per = self.mirrored_strategy.run(self.train_gen_step, args=(imgs_lr, imgs_hr,))
        total_loss = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, total_loss_per,axis=None)
        return total_loss
 

    def compile_generator(self, model):
        """Compile the generator with appropriate optimizer"""
        def pixel_loss(y_true, y_pred):
             #loss1=tf.losses.mean_squared_error(y_true,y_pred)
             loss2=tf.compat.v1.losses.absolute_difference(y_true,y_pred)
             return loss2
             #return 1*loss1+0.005*loss2
        def mae_loss(y_true, y_pred):
            mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

            loss=mae(y_true,y_pred)
            return loss
        def grad_loss(y_true,y_pred):
            grad_hr_1 = tf.compat.v1.py_func(grad1,[y_true],tf.float32)
            grad_hr_2 = tf.compat.v1.py_func(grad2,[y_true],tf.float32)
            grad_hr_3 = tf.compat.v1.py_func(grad3,[y_true],tf.float32)
            grad_sr_1 = tf.compat.v1.py_func(grad1,[y_pred],tf.float32)
            grad_sr_2 = tf.compat.v1.py_func(grad2,[y_pred],tf.float32)
            grad_sr_3 = tf.compat.v1.py_func(grad3,[y_pred],tf.float32)

            mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

            grad_loss =  mse(grad_hr_1,grad_sr_1) +\
                mse(grad_hr_2,grad_sr_2) +\
                mse(grad_hr_3,grad_sr_3)
            loss2 = mse(y_true,y_pred)
            return grad_loss + loss2
            #return loss2
        def grad_loss2(y_true, y_pred):
            #grads = tf.nn.compute_average_loss(tf.py_function(grad_all,[y_true, y_pred], tf.float32))

            mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            loss2 = tf.nn.compute_average_loss(mse(y_true,y_pred))

            #grads = tf.compat.v1.py_func(grad_all,[y_true, y_pred], tf.float32)
            #loss2 = tf.keras.losses.mean_squared_error(y_true, y_pred)
            #return tf.nn.compute_average_loss(grads + loss2)
            #mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
            #loss = tf.nn.compute_average_loss(mae(y_true, y_pred))
            #return loss
            return loss2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.gen_lr,beta_1=0.9,beta_2=0.999)
        #self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.gen_lr)
        #optimizer = hvd.DistributedOptimizer(optimizer)
        model.compile(
            loss=grad_loss,
            optimizer=self.optimizer,
            metrics=['mse'],
            #experimental_run_tf_function=False
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
        
            
#here starts python execution commands
# Run the PIESRGAN network
if __name__ == '__main__':
   
    t1 = time.time()
    print(">> Creating the PIESRGAN network")
    gan = PIESRGAN(training_mode=True,
                height_lr = 32, width_lr=32, depth_lr=32,
                gen_lr=1e-6, dis_lr=5e-4,
                channels=1
                )
    # # Stage1: Train the generator w.r.t RRDB first
    print(">> Start training generator")
    print(">> training [ts=5500]")
    print("GAN creation took {} secs".format(time.time()-t1))
    gan.generator.load_weights('./data/weights/generator_idx_1600.h5')
    gan.generator.save_weights('./data/weights/test_idx_1600.h5')
    #gan.train_generator(
    #     epochs=10,
    #     datapath_train='../../../hpcwork/zl963564/box5500.h5',
    #     batch_size=32,
    #)
# Deepak
    #gan.load_weights(generator_weights='./data/weights4/_generator_idx280.h5')
    #gan.train_generator(
    ##     epochs=100,
    #     boxsize=16,
    #     datapath_train='/scratch/w47/share/IsotropicTurb',
    #     datapath_test='/scratch/w47/share/IsotropicTurb',
    #     batch_size=32,
    #     steps_per_validation=1,
    #     steps_per_epoch=1,
    #     workers=1,
    #     log_weight_path='./data/weights/',
    #     use_multiprocessing=True
    #)

    #gan.load_weights(generator_weights='./data/weights2/_generator_idx1300.h5')

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
    #     epochs=3000,
    #     first_epoch=0,
    #     batch_size=32,
    #     boxsize=16,
    #     dataname='IsoTurb',
    #     log_weight_path='./data/weights4/',
    #   #datapath_train='../datasets/DIV2K_224/',
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

