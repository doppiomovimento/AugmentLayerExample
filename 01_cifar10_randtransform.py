# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import tensorflow as tf
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K


def implt(img):
    plt.figure()
    plt.imshow(img)
    plt.axis('off')




                            # Set up 'ggplot' style
plt.style.use('ggplot')     # if want to use the default style, set 'classic'
plt.rcParams['ytick.right']     = True
plt.rcParams['ytick.labelright']= True
plt.rcParams['ytick.left']      = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['font.family']     = 'Arial'


# .............................................................................


data            = cifar10.load_data()
(trdata,trlabel)= data[0]
(tsDat, tsLbl)  = data[1]



                            # Convert the data into 'float32'
                            # Rescale the values from 0~255 to 0~1
trdata      = trdata.astype('float32')/255
tsDat       = tsDat.astype('float32')/255


                            # Retrieve the row size of each image
                            # Retrieve the column size of each image
imgrows     = trdata.shape[1]
imgclms     = trdata.shape[2]
channel     = trdata.shape[3]


                            # Perform one hot encoding on the labels
                            # Retrieve the number of classes in this problem
trlabel     = to_categorical(trlabel)
tsLbl       = to_categorical(tsLbl)
num_classes = tsLbl.shape[1]


                            # Split the original training dataset into
                            # one dataset for training, another for validation
trDat       = trdata[0:40000]
trLbl       = trlabel[0:40000]
vlDat       = trdata[40000:50000]
vlLbl       = trlabel[40000:50000]



# .............................................................................

                            # fix random seed for reproducibility
seed        = 29
np.random.seed(seed)



def augmentLayer(inputs,
                 rot90=False, 
                 flipUpDown=False, 
                 flipLeftRight=False,
                 gamma=None,              # To use, say, gamma=[0.5,1.5]
                                             # above 1, shift histogram to left; below 1, shift histogram to right
                 seed=None):
    
    if inputs.dtype != tf.float32:
            inputs = tf.image.convert_image_dtype(inputs, 
                                                  dtype=tf.float32)
    def tfImgTransform(inputs):
        def tfRot90(x):
            rand        = tf.random.uniform(shape=[],       # '[]' returns a scalar instead of a tensor
                                            maxval=4,       # the setup is [minval, maxval), hence maxval is 4, not 3
                                            dtype=tf.dtypes.int32,
                                            seed=seed)
            x           = tf.image.rot90(x,
                                         k=rand)
            return x
        def tfGamma(x):
            rand        = tf.random.uniform(shape=[],       # '[]' returns a scalar instead of a tensor
                                            minval=gamma[0],
                                            maxval=gamma[1],
                                            dtype=tf.dtypes.float32,
                                            seed=seed)
            x           = tf.image.adjust_gamma(x,
                                                gamma=rand,
                                                gain=1)
            return x
        
        if flipUpDown:
            inputs      = tf.image.random_flip_up_down(inputs, 
                                                       seed=seed)
        if flipLeftRight:
            inputs      = tf.image.random_flip_left_right(inputs, 
                                                          seed=seed)        
        if rot90:
            inputs      = tf.map_fn(tfRot90,inputs)
            
        if gamma:
            inputs      = tf.map_fn(tfGamma,inputs)
        
        return inputs
    
    return K.in_train_phase(tfImgTransform(inputs), inputs)




optmz       = optimizers.RMSprop(lr=0.0001, decay=1e-6)

modelname   = '01_cifar10'
                            # define the deep learning model



def createModel():
    ipt     = Input(shape=(imgrows,imgclms,channel))
    lb      = Lambda(augmentLayer,
                     input_shape=(imgrows,imgclms,channel),
                     output_shape=(imgrows,imgclms,channel),
                     arguments={'gamma': [0.2,1.8],
                                'flipLeftRight': True,
                                'seed': seed})(ipt)
    
    x       = Conv2D(32,(3,3),padding='same',activation='relu')(lb)
    x       = Conv2D(32,(3,3),padding='same',activation='relu')(x)
    x       = MaxPooling2D(pool_size=(2,2))(x)
    x       = Dropout(0.25)(x)
        
    x       = Conv2D(64,(3,3),padding='same',activation='relu')(x)
    x       = Conv2D(64,(3,3),padding='same',activation='relu')(x)
    x       = MaxPooling2D(pool_size=(2,2))(x)
    x       = Dropout(0.25)(x)
    
    x       = Flatten()(x)
    x       = Dense(512,activation='relu')(x)
    x       = Dropout(0.5)(x)
    x       = Dense(num_classes,activation='softmax')(x)
    
    model   = Model(inputs=ipt,outputs=x)
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer=optmz, 
                  metrics=['accuracy'])
    
    return model





                            # Setup the models
model       = createModel() # This is meant for training
modelGo     = createModel() # This is used for final testing

model.summary()



# .............................................................................


                            # Create checkpoint for the training
                            # This checkpoint performs model saving when
                            # an epoch gives highest testing accuracy
filepath        = modelname + ".hdf5"
checkpoint      = ModelCheckpoint(filepath, 
                                  monitor='val_loss', 
                                  verbose=0, 
                                  save_best_only=True, 
                                  mode='min')

                            # Log the epoch detail into csv
csv_logger      = CSVLogger(modelname +'.csv')
callbacks_list  = [checkpoint,csv_logger]



# .............................................................................


                            # Fit the model
                            # This is where the training starts
model.fit(trDat, 
          trLbl, 
          validation_data=(vlDat, vlLbl), 
          epochs=100,       
          batch_size=128,
          shuffle=True,
          callbacks=callbacks_list)



# ......................................................................


                            # Now the training is complete, we get
                            # another object to load the weights
                            # compile it, so that we can do 
                            # final evaluation on it
modelGo.load_weights(filepath)
modelGo.compile(loss='categorical_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])

 

                            # Get the best accuracy on the validation dataset
validScores = modelGo.evaluate(vlDat, 
                               vlLbl, 
                               verbose=0)
print("Best accuracy (on validation dataset): %.2f%%" % (validScores[1]*100))




# .......................................................................


                            # Make classification on the test dataset
predicts    = modelGo.predict(tsDat)


                            # Prepare the classification output
                            # for the classification report
predout     = np.argmax(predicts,axis=1)
testout     = np.argmax(tsLbl,axis=1)
labelname   = ['airplane',
               'automobile',
               'bird',
               'cat',
               'deer',
               'dog',
               'frog',
               'horse',
               'ship',
               'truck']
                                            # the labels for the classfication report


testScores  = metrics.accuracy_score(testout,predout)
confusion   = metrics.confusion_matrix(testout,predout)


print("Best accuracy (on testing dataset): %.2f%%" % (testScores*100))
print(metrics.classification_report(testout,predout,target_names=labelname,digits=4))
print(confusion)


    
    
    
# ..................................................................
    
import pandas as pd

records     = pd.read_csv(modelname +'.csv')
plt.figure()
plt.subplot(211)
plt.plot(records['val_loss'])
plt.yticks([0.00,0.60,0.70,0.80])
plt.title('Loss value',fontsize=12)

ax          = plt.gca()
ax.set_xticklabels([])



plt.subplot(212)
plt.plot(records['val_acc'])
plt.yticks([0.5,0.6,0.7,0.8])
plt.title('Accuracy',fontsize=12)
plt.show()

# ..................................................................


                                            # This way of getting layer output
                                            # is only working in graph mode
                                            # In eager execution, this will give
                                            # error!!!
getLayerOutput  = K.function([model.layers[0].input,K.learning_phase()],
                             [model.layers[1].output])

lyrOutput       = getLayerOutput([trdata[0:10],1])[0]

                                            # the '1' after 'x_test[0:10]' stands for
                                            # output in learning phase
                                            # Use '0' to get the output in testing phase