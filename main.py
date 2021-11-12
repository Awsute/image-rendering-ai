from io import BytesIO
import json
import numpy as np
import neuralnet as nn
import random
from png import makeGrayPNG
import training_data

img_size = [256, 256]

drw_model = nn.Network(
    1,
    [], 
    nn.Activator(lambda x: nn.safe_sigmoid(x), lambda x: nn.safe_sigmoid(x)*(1-nn.safe_sigmoid(x)))
)

int_model = nn.Network(
    img_size[0]*img_size[1], 
    [], 
    nn.Activator(lambda x: nn.safe_sigmoid(x), lambda x: nn.safe_sigmoid(x)*(1-nn.safe_sigmoid(x)))
)


drw_model.hidden = drw_model.random_net(16, 64, img_size[0]*img_size[1])
#model.import_from_file("model.json")

input = [random.random()]
data = drw_model.predict(input)


#this is for rendering image
data = data[0][len(data[0])-1]
d = []
r = 0
row = []
for i in range(len(data)):
    if (i+1)%img_size[0] == 0:
        d.append(row)
        row = []
        r += 1
    else:
        row.append(int(255*data[i]))
print(r)
with open("img.png","wb") as f:
    f.write(makeGrayPNG(d, img_size[1], img_size[0]))

#model.output_to_file("model.json")