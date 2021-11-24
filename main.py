from io import BytesIO
import json
import numpy as np
import src.neuralnet as nn
import random
import src.graphics as g
import src.ppm
import src.interpreter
from src.renerer import *
from src.gradients import gradient, normalize
import random



render_model = Network(
    src.interpreter.TEXT_VECTOR_LEN,
    [], 
    Activator(lambda x: safe_sigmoid(x), lambda x: safe_sigmoid(x)*(1-safe_sigmoid(x)))
)
render_model.use_bias = 1.0
render_model.hidden = render_model.random_net(3, 3, 3*IMG_SIZE_INITIAL[0]*IMG_SIZE_INITIAL[1])
#render_model.import_from_file("assets/draw_model.json")
input = [random.random()]*src.interpreter.TEXT_VECTOR_LEN
data, a = render_model.predict(input)
#this is for rendering image
data = data[len(data)-1]
def render_data(data, file):
    d = []
    for i in range(int(len(data))):
        d.append(int(data[i]*255))
    src.ppm.create_image(IMG_SIZE_INITIAL, d, file)

render_data(data, "img.ppm")
grad = normalize(gradient(IMG_SIZE_INITIAL, [0, 0, 0], [255, 255, 255], 0))


for i in range(0, 10000):
    render_model.hidden = render_model.backprop([0, 1, 0]*IMG_SIZE_INITIAL[0]*IMG_SIZE_INITIAL[1], input, 0.001)
    data, a = render_model.predict(input)
    data = data[len(data)-1]
    print(data)

render_data(data, "img1.ppm")
render_model.output_to_file("assets/draw_model.json")