from data_processor import load_and_preprocess_image
from visualizer import visualizer, analysis
from sem_model import sem_model
import tensorflow as tf
import glob
import numpy as np
import os
import config


AUTOTUNE = tf.data.experimental.AUTOTUNE
os.chdir(config.base_dir)
train_dir = './test_images'

image_height = config.image_height
image_width = config.image_width
img_size = config.img_size
num_img = config.num_img

train_label = np.genfromtxt(train_dir + '/label.csv', delimiter=',')
Rx = img_size / image_width
Ry = img_size / image_height
for i in train_label:
    i[0::3] *= Rx
    i[1::3] *= Ry
print('train_label:', train_label.shape)

train_img_paths = sorted(glob.glob(train_dir + '/*.jpg'))
path_ds = tf.data.Dataset.from_tensor_slices(train_img_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

train_data = np.zeros((num_img, img_size, img_size, 1))
for i, img in enumerate(image_ds):
    train_data[i] = img

print('train_data:', train_data.shape)

model = sem_model(train_data, train_label)


# pred = model.predict(train_data)
