from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import os
import pickle
import numpy as np
import cv2
from PIL import Image

# def get_fcn_model():
#     tf.keras.backend.clear_session()
#     model = models.Sequential(name = "FCN_Implementation")

#     #Encoder
#     layer1 = layers.Conv2D(96, (3, 3),input_shape = (224,224,3) , name = "Encoding_layer_1")
#     layer2 = layers.Conv2D(128, (3, 3), name = "Encoding_layer_2" )
#     layer3 = layers.Conv2D(192, (3, 3), name = "Encoding_layer_3" )
#     #layer4 = layers.Conv2D(384, (3, 3), name = "Encoding_layer_4" )
#     #layer5 = layers.Conv2D(128, (3, 3), name = "Encoding_layer_5" )
#     #layer6 = layers.Conv2D(1024, (3, 3), name = "Encoding_layer_6")
#     #layer7 = layers.Conv2D(4096, (3, 3), name = "Encoding_layer_7")
#     layer8 = layers.Conv2D(1, (7, 7),name = "Encoding_layer_8")

#     #Decoder
#     layer9 = layers.Conv2DTranspose(91,(7,7), name = "Decoding_layer_1")
#     #layer10 = layers.Conv2DTranspose(4096,(3,3), name = "Decoding_layer_2")
#     #layer11 = layers.Conv2DTranspose(1024,(3,3), name = "Decoding_layer_3")
#     #layer12 = layers.Conv2DTranspose(128,(3,3), name = "Decoding_layer_4")
#     #layer13 = layers.Conv2DTranspose(384,(3,3), name = "Decoding_layer_5")
#     layer14 = layers.Conv2DTranspose(192,(3,3), name = "Decoding_layer_6")
#     layer15 = layers.Conv2DTranspose(128,(3,3), name = "Decoding_layer_7")
#     layer16 = layers.Conv2DTranspose(91,(3,3), name = "Decoding_layer_8")

#     #Make Model
#     model.add(layer1)
#     model.add(layer2)
#     model.add(layer3)
#     #model.add(layer4)
#     #model.add(layer5)
#     #model.add(layer6)
#     #model.add(layer7)
#     model.add(layer8)

#     model.add(layer9)
#     #model.add(layer10)
#     #model.add(layer11)
#     #model.add(layer12)
#     #model.add(layer13)
#     model.add(layer14)
#     model.add(layer15)
#     model.add(layer16)
    
    
    
#     model.build((None, 224, 224, 3))
#     model.compile(optimizer = 'adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'],verbose = 2)
#     return model

def get_unet_model(dropout=0.05,n_filters = 16):
    tf.keras.backend.clear_session()
    model = models.Sequential(name = "UNet_Implementation")
    #base_model = tf.keras.applications.MobileNetV2(input_shape=[224, 224, 3], include_top=False)
    
    #Contractive Path
    layer1 = layers.Conv2D(n_filters, (3, 3),input_shape = (224,224,3) ,activation = tf.nn.leaky_relu,name = "Encoding_layer_1")
    layer2 = layers.BatchNormalization()
    layer3 = layers.MaxPool2D(pool_size=(2, 2))
    layer4 = layers.Dropout(dropout*0.5)
    
    layer5 = layers.Conv2D(n_filters*2, (3, 3) ,activation = tf.nn.leaky_relu,name = "Encoding_layer_2")
    layer6 = layers.BatchNormalization()
    layer7 = layers.MaxPool2D(pool_size=(2, 2))
    layer8 = layers.Dropout(dropout*0.5)
    
    layer9 = layers.Conv2D(n_filters*4, (3, 3) ,activation = tf.nn.leaky_relu,name = "Encoding_layer_3")
    layer10 = layers.BatchNormalization()
    layer11 = layers.MaxPool2D(pool_size=(2, 2))
    layer12 = layers.Dropout(dropout*0.5)
    
    layer13 = layers.Conv2D(n_filters*8, (3, 3) ,activation = tf.nn.leaky_relu,name = "Encoding_layer_4")
    layer14 = layers.BatchNormalization()
    layer15 = layers.MaxPool2D(pool_size=(2, 2))
    layer16 = layers.Dropout(dropout*0.5)
    
    #Expansive Path
    #layer18 = tf.concat([layer17,layer13],axis = -1)
    
    layer17 = layers.Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same')
    layer18 = layers.Dropout(dropout*0.5)
    #layer19 = layers.Conv2D(n_filters*8, (3, 3) ,activation = tf.nn.leaky_relu)
    layer20 = layers.BatchNormalization()
    
    layer21 = layers.Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same')
    layer22 = layers.Dropout(dropout*0.5)
    #layer23 = layers.Conv2D(n_filters*4, (3, 3) ,activation = tf.nn.leaky_relu)
    layer24 = layers.BatchNormalization()
    
    layer25 = layers.Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same')
    layer26 = layers.Dropout(dropout*0.5)
    #layer27 = layers.Conv2D(n_filters*2, (3, 3) ,activation = tf.nn.leaky_relu)
    layer28 = layers.BatchNormalization()
    
    layer29 = layers.Conv2DTranspose(n_filters*1, (3, 3), strides=(3, 3), padding='same')
    layer30 = layers.Dropout(dropout*0.5)
    layer31 = layers.Conv2D(n_filters*1, (65, 65) ,activation = tf.nn.leaky_relu)
    layer32 = layers.BatchNormalization()
    
    layer33 = layers.Conv2D(1, (1, 1) ,activation = tf.nn.sigmoid)
    
    model.add(layer1)
    model.add(layer2)
    model.add(layer3)
    model.add(layer4)
    model.add(layer5)
    model.add(layer6)
    model.add(layer7)
    model.add(layer8)
    model.add(layer9)
    model.add(layer10)
    model.add(layer11)
    model.add(layer12)
    model.add(layer13)
    model.add(layer14)
    model.add(layer15)
    model.add(layer16)
    model.add(layer17)
    model.add(layer18)
    #model.add(layer19)
    model.add(layer20)
    model.add(layer21)
    model.add(layer22)
    #model.add(layer23)
    model.add(layer24)
    model.add(layer25)
    model.add(layer26)
    #model.add(layer27)
    model.add(layer28)
    model.add(layer29)
    model.add(layer30)
    model.add(layer31)
    model.add(layer32)
    model.add(layer33)

    model.compile(optimizer = 'adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics = ['accuracy'],verbose = 2)
    return model    


def pickle_save(variable,file_name):
    with open(file_name,"wb") as saveFile:
        pickle.dumb(variable,file_name)
    
def pickle_load(file_name):
    with open(file_name,"rb") as saveFile:
        return pickle.load( saveFile)
    
def pre_process(numpy_array):
    #numpy_array[numpy_array >= 0.5] = 1
    #numpy_array[numpy_array < 0.5] = 0
    #numpy_array = numpy_array.astype('int8')
    numpy_array = numpy_array*255
    return numpy_array

def save_image(numpy_array,file_name):
    if (np.max(numpy_array)<=1):
        numpy_array = pre_process(numpy_array)
    cv2.imwrite(file_name,numpy_array)
    
def generate_tests(train_x,train_y,count,random_image_index = None):
    if random_image_index==None:
        random_image_index = np.random.randint(low=0, high=len(train_x), size=count)
    output_x = list()
    output_y = list()
    output_names = list()
    for index in random_image_index:
        output_x.append(train_x[index])
        output_y.append(train_y[index])
        output_names.append("Id - " + str(index))
    return np.asarray(output_x),np.asarray(output_y),output_names

def save_images(image_list,image_names,prepend="",postpend=""):
    for i in range(len(image_list)):
        img = image_list[i]
        img_name = image_names[i]
        img_name = prepend+img_name+ postpend
        save_image(img,img_name)

def save_model(model,path):
    tf.saved_model.save(model, path)
    
def load_model(path):
    return tf.saved_model.load(path)

def predict_y(model,image_batch):
    image_batch = image_batch.astype('float32')
    return model.predict(image_batch)
#     shape = image_batch.shape
#     image_batch = image_batch.reshape((shape[0],shape[1],shape[2],1))
#     output = list()
#     for image in image_batch:
#         predicted = model.predict(image)
#         output.append(output)
#     return np.asarray(output)

def single_file_read(input_path,width,height):
    img1 = Image.open(input_path)
    output = np.asarray(img1)
    try:
        if output.shape[2]==4:
            output = output[:,:,:3]
    except:
        pass
    output = cv2.resize(output, (width, height))
    return output

def threshold_images(image_batch,thresh = 0.5):
    output = list()
    for image in image_batch:
        image[image >= 0.5] = 1
        image[image < 0.5] = 0
        image = image.astype('int8')
        image = image*255
        output.append(image)
    return np.asarray(output)

