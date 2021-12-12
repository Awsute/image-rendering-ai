#How renderer is going to work:
#Use a reverse cnn
#Since the text vector dimension count is a square, it can be turned into a square for rendering
#Input text vector (paragraph vector) and output an image with a relatively small size (like 64x64)
#Use a neural network to upscale the quality of the image

#Training:
#Get text vector and correct image and train based on the incorrect output
import src.interpreter
import math
import tensorflow as tf
IMG_SIZE_INITIAL = 64
IMG_SIZE_FINAL = 256

def setup_renderer_initial():
    i_shape = src.interpreter.TEXT_VECTOR_LEN
    initializer = tf.keras.initializers.HeUniform()
    a = [
        tf.keras.layers.Input(shape=(i_shape,)),
    ]
    
    for i in range(8):
        a.append(tf.keras.layers.Dense(8, activation='relu', kernel_initializer=initializer))
    a.append(tf.keras.layers.Dense(3*IMG_SIZE_INITIAL**2, activation='sigmoid', kernel_initializer=initializer))
    model = tf.keras.Sequential(a)
    return model

def setup_renderer_secondary():
    i_shape = 3*IMG_SIZE_INITIAL**2
    initializer = tf.keras.initializers.HeUniform()
    a = [
        tf.keras.layers.Input(shape=(i_shape,)),
    ]
    
    for i in range(8):
        a.append(tf.keras.layers.Dense(16, activation='relu', kernel_initializer=initializer))
    a.append(tf.keras.layers.Dense(3*IMG_SIZE_FINAL**2, activation='sigmoid', kernel_initializer=initializer))
    model = tf.keras.Sequential(a)
    return model
