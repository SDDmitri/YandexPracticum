#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.optimizers import Adam


# In[2]:


RANDOM_STATE = 270923


# In[3]:


def load_train(path):
    train_datagen = ImageDataGenerator(validation_split=0.25,
                                   horizontal_flip=True,
                                   rotation_range=45,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   rescale=1/255.,
                                   zoom_range=0.2
                                   )
    
    train_datagen_flow = train_datagen.flow_from_dataframe(
        dataframe= pd.read_csv(path+'/labels.csv'),
        directory=path+'/final_files',
        x_col='file_name',
        y_col='real_age',
        batch_size=16,
        class_mode='raw',
        seed=RANDOM_STATE,
        subset='training',
        target_size=(224, 224)
        )
    return train_datagen_flow


# In[4]:


def load_test(path):
    test_datagen = ImageDataGenerator(validation_split=0.25,                                   
                                      rescale=1/255.
                                      )
    
    test_datagen_flow = test_datagen.flow_from_dataframe(       
        dataframe= pd.read_csv(path+'/labels.csv'),
        directory=path+'/final_files',
        x_col='file_name',
        y_col='real_age',
        batch_size=16,
        class_mode='raw',
        seed=RANDOM_STATE,
        subset='validation',
        target_size=(224, 224)
        )
    return test_datagen_flow


# In[5]:


def create_model(input_shape):
    backbone = ResNet50(input_shape=input_shape,
                        weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',						
                        include_top=False)     
    model = Sequential()   
    model.add(backbone)
    model.add(GlobalAveragePooling2D())    
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='relu'))
    optimizer = Adam(learning_rate=1e-5)
    model.summary()
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model


# In[6]:


def train_model(model, train_data, test_data, batch_size=None, epochs=20,
               steps_per_epoch=None, validation_steps=None):               

    model.fit(train_data, 
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)
    return model

