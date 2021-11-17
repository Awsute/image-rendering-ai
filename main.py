from io import BytesIO
import src
import json
import numpy as np
import src.neuralnet as nn
import random
import src.graphics as g
import src.ppm
import src.interpreter
import src.renerer
from src.gradients import gradient, normalize


img_size = [256, 256]

render_model = nn.Network(
    src.interpreter.TEXT_VECTOR_LEN,
    [], 
    nn.Activator(lambda x: nn.relu(x), lambda x: nn.drelu(x))
)
render_model.use_bias = 0.0
render_model.hidden = render_model.random_net(16, 8, 3*img_size[0]*img_size[1])
#render_model.import_from_file("assets/draw_model.json")
input = [random.random()]*src.interpreter.TEXT_VECTOR_LEN
data, a = render_model.predict(input)
#this is for rendering image
data = data[len(data)-1]
def render_data(data, file):
    d = []
    for i in range(int(len(data))):
        d.append(int(data[i]*255))
    src.ppm.create_image(img_size[0], img_size[1], d, file)

render_data(data, "img.ppm")
grad = normalize(gradient(img_size[0], img_size[1], [0, 0, 40], [255, 255, 255], 0))

render_model.backprop(grad, input, 0.1)

data, a = render_model.predict(input)

#this is for rendering image
data = data[len(data)-1]
print(data)
render_data(data, "img1.ppm")
render_model.output_to_file("assets/draw_model.json")