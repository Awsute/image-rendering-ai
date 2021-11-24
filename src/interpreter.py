import src.neuralnet as nn

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


def process_text(text):
    text = text.lower()
    for i in text:
        if VALID_CHARS.find(i) < 0:
            text = text.replace(i, "")
    return text
def vector_of_text(text : str, word_model : nn.Network, text_model : nn.Network):
    text = process_text(text)
    words = text.split(" ")
    vec_t = [0.0]*TEXT_VECTOR_LEN
    for word in words:
        vec_w = [0.0]*WORD_VECTOR_LEN
        for letter in word:
            input = vec_w
            input.append(float(ord(letter)))
            n_v, a = word_model.predict(input)
            vec_w = n_v[len(n_v)-1][:len(n_v[len(n_v)-1])-1]
        input = vec_t
        input.extend(vec_w)
        n_v, a = text_model.predict(input)
        vec_t = n_v[len(n_v)-1][:len(n_v[len(n_v)-1])-WORD_VECTOR_LEN]
    return vec_t


