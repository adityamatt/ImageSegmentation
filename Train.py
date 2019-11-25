# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from PIL import Image
from tensorflow.keras import datasets, layers, models
from util import *
import cv2

#import matplotlib.pyplot as plt

"""
OLD CODE
# ##Training Data
# new_train_data,new_test_data = get_train_test_data(train_folder,test_folder)
# new_test_data = np.expand_dims(new_test_data,-1)
# print(new_train_data.shape,new_test_data.shape)
"""
print("---------------------------------------------------------------------------------------------")
print("Loading Data")
training_data_dir = "./Vessel/DATA.TXT"
train_x,train_y = pickle_load(training_data_dir)
print(train_x.shape,train_y.shape)

print("Loading Model")
model = get_unet_model()
model.summary()

#Check points for training
checkpoint_path = "./Model/cp.ckpt"
checkpoint_dir  = os.path.dirname(checkpoint_path)
cp_callback     = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only = True, verbose = 1)

test_x,test_y, test_names = generate_tests(train_x,train_y,10,[153,198,257,27,325,39,399,40,421,422])

predicted_y_initial = predict_y(model,test_x)
loss,acc = model.evaluate(test_x,test_y)
print("Initial Loss and Accuracy:",loss,acc)
model.fit(train_x,train_y,batch_size=32, epochs=20,callbacks = [cp_callback])
loss,acc = model.evaluate(test_x,test_y)
print("Final Loss and Accuracy:",loss,acc)

predicted_y_final = predict_y(model,test_x)
save_images(test_x,test_names,"./Data/","-Test-X.jpg")
save_images(test_y,test_names,"./Data/","-Test-Y.jpg")
save_images(predicted_y_initial,test_names,"./Data/","-Predicted-Y-initial.jpg")
save_images(predicted_y_final,test_names,"./Data/","-Predicted-Y-Final.jpg")


# train_folder = "./MS-COCO/Train/"
# test_folder = "./MS-COCO/Test/"





#print(new_train_data[0])


#tmp = model.fit(new_train_data,new_test_data,batch_size= 32, epochs =2,verbose = 2)









