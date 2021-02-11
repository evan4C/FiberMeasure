import glob
import numpy as np
import csv
import tensorflow as tf
import config


def data_processor(data_dir):
    """
    convert all csv into one csv file named 'label.csv'
    """
    output = data_dir + '/label.csv'
    with open(output, 'w', newline='') as f:
        writer = csv.writer(f)
        point_files = sorted(glob.glob(data_dir + '/*000000.csv'))
        for file in point_files:
            points = np.loadtxt(file, skiprows=1, delimiter=',')
            points = points.flatten()[:15]
            writer.writerow(points)


def preprocess_image(image, img_size):
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [img_size, img_size])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    img_size = config.img_size
    return preprocess_image(image, img_size)


if __name__ == '__main__':
    data_dir = ''
    data_processor(data_dir)
