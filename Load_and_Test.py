from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from PIL import Image
from tensorflow.keras import datasets, layers, models
from util import *
import cv2
import matplotlib.pyplot as plt
from PIL import Image

print("---------------------------------------------------------------------------------------------")
print("Loading Data")
training_data_dir = "./Vessel/DATA.TXT"
train_x,train_y = pickle_load(training_data_dir)
print(train_x.shape,train_y.shape)

print("Loading Model")
model = get_unet_model()

#Check points for training
checkpoint_path = "./Model/cp.ckpt"
checkpoint_dir  = os.path.dirname(checkpoint_path)
cp_callback     = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only = True, verbose = 1)

#Load weights
model.load_weights(checkpoint_path)
#Loss from terminal
loss = [0.8137,0.8060,0.6180,0.5721,0.5301, 0.4788, 0.4590, 0.4370, 0.4139, 0.4022, 0.3872, 0.3825, 0.3694, 0.3592, 0.3446, 0.3253, 0.3158, 0.3160, 0.3084, 0.3046]
accuracy = [0.6161,0.6978,0.7266, 0.7527, 0.7685, 0.7845, 0.7956, 0.8063, 0.8173, 0.8241,0.8320,0.8343, 0.8402, 0.8447, 0.8529, 0.8617, 0.8665, 0.8663, 0.8711, 0.8719]
epoch = list(range(1,21,1))

# plt.plot(epoch,loss,label = 'loss')
# plt.plot(epoch,accuracy, label ='accuracy')
# plt.legend()
# plt.savefig("loss_acc.png")
##################################################################################
test_image = "./plane.jpg"
test_input = np.asarray([single_file_read(test_image,224,224)])
predicted_output = predict_y(model,test_input)
threshold_images(predicted_output,0.5)
save_images(predicted_output,["Test_output"],"",".png")
