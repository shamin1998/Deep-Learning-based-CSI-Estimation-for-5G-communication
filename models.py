from keras.layers import Convolution2D,Input,BatchNormalization,Conv2D,Activation,Subtract
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from scipy import interpolate
import numpy as np
import math

def psnr(target, ref):
    # Assume image is BGR
    diff = np.array(ref, dtype=float)-np.array(target, dtype=float)
    diff = diff.flatten('C')

    irmse = 1/math.sqrt(np.mean(diff ** 2.0))

    return math.log10(irmse*255.0)*20

def interpolation(noisy , SNR , Number_of_pilot , interp)
    noisy_image = np.zeros((40000,72,14,2))

    noisy_image[:,:,:,0] = np.real(noisy)
    noisy_image[:,:,:,1] = np.imag(noisy)


    if (Number_of_pilot == 48):
        idx = [14*i for i in range(1, 72,6)]+[4+14*(i) for i in range(4, 72,6)]+[7+14*(i) for i in range(1, 72,6)]+[11+14*(i) for i in range(4, 72,6)]
    elif (Number_of_pilot == 16):
        idx= [4+14*(i) for i in range(1, 72,9)]+[9+14*(i) for i in range(4, 72,9)]
    elif (Number_of_pilot == 24):
        idx = [14*i for i in range(1,72,9)]+ [6+14*i for i in range(4,72,9)]+ [11+14*i for i in range(1,72,9)]
    elif (Number_of_pilot == 8):
      idx = [4+14*(i) for  i in range(5,72,18)]+[9+14*(i) for i in range(8,72,18)]
    elif (Number_of_pilot == 36):
      idx = [14*(i) for  i in range(1,72,6)]+[6+14*(i) for i in range(4,72,6)] + [11+14*i for i in range(1,72,6)]



    r = [x//14 for x in idx]
    c = [x%14 for x in idx]



    interp_noisy = np.zeros((40000,72,14,2))

    for i in range(len(noisy)):
        z = [noisy_image[i,j,k,0] for j,k in zip(r,c)]
        if(interp == 'rbf'):
            f = interpolate.Rbf(np.array(r).astype(float), np.array(c).astype(float), z,function='gaussian')
            X , Y = np.meshgrid(range(72),range(14))
            z_intp = f(X, Y)
            interp_noisy[i,:,:,0] = z_intp.T
        elif(interp == 'spline'):
            tck = interpolate.bisplrep(np.array(r).astype(float), np.array(c).astype(float), z)
            z_intp = interpolate.bisplev(range(72),range(14),tck)
            interp_noisy[i,:,:,0] = z_intp
        z = [noisy_image[i,j,k,1] for j,k in zip(r,c)]
        if(interp == 'rbf'):
            f = interpolate.Rbf(np.array(r).astype(float), np.array(c).astype(float), z,function='gaussian')
            X , Y = np.meshgrid(range(72),range(14))
            z_intp = f(X, Y)
            interp_noisy[i,:,:,1] = z_intp.T
        elif(interp == 'spline'):
            tck = interpolate.bisplrep(np.array(r).astype(float), np.array(c).astype(float), z)
            z_intp = interpolate.bisplev(range(72),range(14),tck)
            interp_noisy[i,:,:,1] = z_intp


    interp_noisy = np.concatenate((interp_noisy[:,:,:,0], interp_noisy[:,:,:,1]), axis=0).reshape(80000, 72, 14, 1)
   
    
    return interp_noisy

def SRCNN_model():
    c1 = Convolution2D( 64 , 9 , 9 , activation = 'relu', init = 'he_normal', border_mode='same')(Input(shape = (72,14,1)))
    c2 = Convolution2D( 32 , 1 , 1 , activation = 'relu', init = 'he_normal', border_mode='same')(c1)
    c3 = Convolution2D( 1 , 5 , 5 , init = 'he_normal', border_mode='same')(c2)
    
    model = Model(input = x, output = c3)
    
    # Compile
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8) 
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error']) 
    return model
    
def SRCNN_train(train_data ,train_label, val_data , val_label , channel_model , num_pilots , SNR ):
    srcnn_model = SRCNN_model()
    print(srcnn_model.summary())
    
    callbacks_list = [ModelCheckpoint("SRCNN_check.h5", monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=False, mode='min')]

    srcnn_model.fit(train_data, train_label, batch_size=128, validation_data=(val_data, val_label),
                    callbacks=callbacks_list, shuffle=True, epochs= 300 , verbose=0)
    
    srcnn_model.save_weights("SRCNN_" + channel_model +"_"+ str(num_pilots) + "_"  + str(SNR) + ".h5")
   


def SRCNN_predict(input_data , channel_model , num_pilots , SNR):
    srcnn_model = SRCNN_model()
    srcnn_model.load_weights("SRCNN_" + channel_model +"_"+ str(num_pilots) + "_"  + str(SNR) + ".h5")
    predicted  = srcnn_model.predict(input_data)
    return predicted

  
def DNCNN_model ():
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(Input(shape=(None,None,1)))
    x = Activation('relu')(x)
    # 18 layers, Conv+BN+relu
    for i in range(18):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)   
    # last layer, Conv
    x = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = Subtract()([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8) 
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])    
    return model

def DNCNN_train(train_data ,train_label, val_data , val_label, channel_model , num_pilots , SNR ):
  
  dncnn_model = DNCNN_model()
  print(dncnn_model.summary())

  callbacks_list = [ModelCheckpoint("DNCNN_check.h5", monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=False, mode='min')]

  dncnn_model.fit(train_data, train_label, batch_size=128, validation_data=(val_data, val_label),
                  callbacks=callbacks_list, shuffle=True, epochs= 200 , verbose=0)
  dncnn_model.save_weights("DNCNN_" + channel_model +"_"+ str(num_pilots) + "_"  + str(SNR) + ".h5")
  
  
  
def DNCNN_predict(input_data, channel_model , num_pilots , SNR):
  dncnn_model = DNCNN_model()
  dncnn_model.load_weights("DNCNN_" + channel_model +"_"+ str(num_pilots) + "_"  + str(SNR) + ".h5")
  predicted  = dncnn_model.predict(input_data)
  return predicted