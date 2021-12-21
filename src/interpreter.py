from typing import Text
import tensorflow as tf
import numpy as np
#how the interpreter is going to work

#STEP 1: recursive nn

#Input a vector (all 0's if at beginning of word) and a character and output a vector
#Input previous vector to next iteration


#STEP 2: a different recursive nn

#Input the vector of the first word and the new vector (all 0's if first word) and return a new vector
#Input new vector to the next iteration

#This method converts a series of words into a vector that computers can understand.

#TRAINING:
#Word model:
#Overall word vector should be approximately the negative vector of a word with opposite meaning.

#Text model:
#Get vector of a sentence with opposite meaning to the input and negate it. 
#This is approximately the vector of the input sentence.




VALID_CHARS = "1234567890abcdefghijklmnopqrstuvwxyz "

WORD_VECTOR_LEN = 9
TEXT_VECTOR_LEN = 16

def setup_word_model():
    initializer = tf.keras.initializers.HeUniform()
    a = [
        tf.keras.layers.Input(shape=(WORD_VECTOR_LEN+1,)),
    ]
    
    for i in range(4):
        a.append(tf.keras.layers.Dense(12, activation='relu', kernel_initializer=initializer))
    a.append(tf.keras.layers.Dense(WORD_VECTOR_LEN, activation='sigmoid', kernel_initializer=initializer))
    model = tf.keras.Sequential(a)
    return model

def setup_text_model():

    initializer = tf.keras.initializers.HeUniform()
    a = [
        tf.keras.layers.Input(shape=(WORD_VECTOR_LEN+TEXT_VECTOR_LEN,)),
    ]
    
    for i in range(4):
        a.append(tf.keras.layers.Dense(12, activation='relu', kernel_initializer=initializer))
    a.append(tf.keras.layers.Dense(TEXT_VECTOR_LEN, activation='sigmoid', kernel_initializer=initializer))
    model = tf.keras.Sequential(a)
    return model

def process_text(text):
    text = text.lower()
    for i in text:
        if VALID_CHARS.find(i) < 0:
            text = text.replace(i, "")
    return text
def vector_of_text(text : str, word_model : tf.keras.Model, text_model : tf.keras.Model):
    text = process_text(text)
    words = text.split(" ")
    vec_t = [0.0]*TEXT_VECTOR_LEN
    for word in words:
        vec_w = [0.0]*WORD_VECTOR_LEN
        for letter in word:
            input = vec_w
            input.append(float(ord(letter)))
            n_v, a = word_model(np.array([input])).numpy()[0]
            lnv = len(n_v)-1
            vec_w = n_v[lnv][:len(n_v[lnv])-1]
        input = vec_t
        input.extend(vec_w)
        n_v, a = text_model(np.array([input])   ).numpy()[0]
        lnv = len(n_v)-1
        vec_t = n_v[lnv][:len(n_v[lnv])-WORD_VECTOR_LEN]
    return vec_t


