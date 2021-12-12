from io import BytesIO
import json
import numpy as np
import random
import png
import tensorflow as tf
import src.graphics as g
import src.ppm
import src.interpreter
from src.renerer import *
from src.gradients import gradient, normalize
import random
import math

data = []


render_model_initial = setup_renderer_initial()
render_model_initial.compile(optimizer="Adam", loss="mse")

render_model_2 = setup_renderer_secondary()
render_model_2.compile(optimizer="Adam", loss="mse")

a = []
for i in range(16):
    a.append(random.random())
a = np.array([a])
data = render_model_initial(a).numpy()

#this is for rendering image
def render_data(data, file, size):
    d = []
    for i in range(int(len(data))):
        d.append(int((data[i]*255)%255))
    src.ppm.create_image(size, d, file)

#STAGE 1: INTERPRET INPUT/TRAIN


#STAGE 2: RENDER LOW RES IMAGE/TRAIN
render_data(data[0], "imgs/img.ppm", IMG_SIZE_INITIAL)
grad_i = normalize(gradient([IMG_SIZE_INITIAL, IMG_SIZE_INITIAL], [255, 255, 0], [255, 0, 255], 45))


render_model_initial.fit(a, np.array([grad_i]), epochs=1000)
data = render_model_initial(a).numpy()
render_data(data[0], "imgs/img1.ppm", IMG_SIZE_INITIAL)

#STAGE 3: CREATE HIGH RES IMAGE/TRAIN
grad_f = normalize(gradient([IMG_SIZE_FINAL, IMG_SIZE_FINAL], [255, 255, 0], [255, 0, 255], 45))
data_f = render_model_2(data).numpy()

render_data(data_f[0], "imgs/img_f.ppm", IMG_SIZE_FINAL)

render_model_2.fit(data, np.array([grad_f]), epochs=1000)

data_f = render_model_2(data).numpy()

render_data(data_f[0], "imgs/img1_f.ppm", IMG_SIZE_FINAL)

