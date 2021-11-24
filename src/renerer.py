#How renderer is going to work:
#Use a reverse cnn
#Since the text vector dimension count is a square, it can be turned into a square for rendering
#Input text vector (paragraph vector) and output an image with a relatively small size (like 64x64)
#Use a neural network to upscale the quality of the image

#Training:
#Get text vector and correct image and train based on the incorrect output
from src.neuralnet import *

IMG_SIZE_INITIAL = [1, 1]
IMG_SIZE_FINAL = [256, 256]