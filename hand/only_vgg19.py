import tensorflow as tf
import numpy as np
import PIL.Image
import tensorflow.keras
import cv2
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D,Input,AveragePooling2D,MaxPooling2D

def add_one_dim(image):
    shape = (1,) + image.shape
    return np.reshape(image, shape)

def load_image(filename, shape=None, max_size=None):
    image = PIL.Image.open(filename)

    if max_size is not None:
        # Calculate the appropriate rescale-factor for
        # ensuring a max height and width, while keeping
        # the proportion between them.
        factor = float(max_size) / np.max(image.size)

        # Scale the image's height and width.
        size = np.array(image.size) * factor

        # The size is now floating-point because it was scaled.
        # But PIL requires the size to be integers.
        size = size.astype(int)

        # Resize the image.
        image = image.resize(size, PIL.Image.LANCZOS) # PIL.Image.LANCZOS is one of resampling filter

    if shape is not None:
        image = image.resize(shape, PIL.Image.LANCZOS) # PIL.Image.LANCZOS is one of resampling filter

    # Convert to numpy floating-point array.
    return np.float32(image)


def vgg_norm():
    img_input = Input(shape=(256, 256, 3))
    x1 = Conv2D(64, (1, 1, 1, 1), activation='relu', padding='same', name='block1_conv1')(img_input)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x1)
    x3 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x2)

    x4 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x3)
    x5 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x4)
    x6 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x5)

    x7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x6)
    x8 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x7)
    x9 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x8)
    x10 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x9)
    x11 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x10)

    x12 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x11)
    x13 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x12)
    x14 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x13)
    x15 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x14)
    x16 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x15)

    x17 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x16)
    x18 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x17)
    x19 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x18)
    x20 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x19)
    x21 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x20)

    model = Model(inputs=[img_input], outputs=[x1, x2, x4, x5, x7, x8, x9, x10, x12, x13, x14, x15])
    model_orig = VGG19(weights='imagenet', input_shape=(256, 256, 3), include_top=False)

    for i in range(len(model.layers)):
        weights = model_orig.layers[i].get_weights()
        model.layers[i].set_weights(weights)

    return model



style_path = "style/wave.jpg"

y_s0 = load_image(style_path)
batch_shape = (4, 256, 256, 3)

# graph input
y_c = tf.placeholder(tf.float32, shape=batch_shape, name='content')
y_s = tf.placeholder(tf.float32, shape=y_s0.shape, name='style')





