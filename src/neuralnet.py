import json
import math
import numpy as np
from random import random, randint
import tensorflow as tf

#NOTES:
#[i1 i2 i3 in 1]
#dot with each:
#	1st hidden layer
#[w11 w21 w31 wn1 b11]
#[w12 w22 w32 wn2 b21]
#[w13 w23 w33 wn3 b23]
#[w1n w2n w3n wnn b2n]
#
#result of dot:
#[h1 h2 h3 hn 1]


#NETWORK HIDDEN:
#network:[
#   for each layer: [
#       for each neuron: [w1 w2 w3 wn b]
#   ]
#]
class Activator:
    def __init__(self, fn, dfn):
        self.fn = fn
        self.dfn = dfn

def safe_sigmoid(x):
    if abs(x) < 700:
        return 1/(1+math.e**-x)
    else:
        return 1

def drelu(x):
    if x < 0:
        return 0
    elif x >= 0:
        return 1
def relu(m):
    if m<0:
        return 0
    else:
        return m

def matrix_softmax(mat):
    m = []
    d = 0
    for i in range(len(mat)):
        d += math.e**mat[i]
        m.append(math.e**mat[i])
    
    for i in range(len(m)):
        m[i] /= d
    
    return m
class Network:
    def __init__(self, input_size, hidden, activator : Activator):
        self.inputs = np.array([0]*input_size)
        self.hidden = hidden
        self.outputs = np.array([])
        self.activator = activator
        self.end_activator = Activator(lambda x: safe_sigmoid(x), lambda x: safe_sigmoid(x)*(1-safe_sigmoid(x)))
        self.use_bias = 0
    
    def random_net(self, hidden_layer_count, layer_size, out_size):
        t = []
        prev_len = len(self.inputs)
        for i in range(hidden_layer_count):
            l = []
            for o in range(layer_size):
                w = []
                for j in range(prev_len + 1):
                    w.append(random()+random()-1)
                l.append(w)
            t.append(l)
            prev_len = layer_size
        l = []
        for i in range(out_size):
            w = []
            for j in range(prev_len+1):
                w.append(random()+random()-1)
            l.append(w)
        t.append(l)
        return np.array(t)
    
    def print_net(self):
        print(self.inputs)
        for layer in self.hidden:
            print(layer)
    
    def predict(self, inputs):
        if len(inputs) != len(self.inputs):
            return
        self.outputs = []
        acs = []
        g = self.hidden
        ins = []
        for i in range(len(inputs)):
            ins.append(self.activator.fn(inputs[i]))
        acs.append(ins)
        self.outputs.append(inputs)
        for l in range(0, len(g)):
            acls = []
            zs = []
            for n in range(0, len(g[l])):
                #print(str(matrix_dot_product(self.outputs[len(self.outputs)-1], g[l][n][0])) + ", " + str(l))
                z = np.dot(np.append(acs[len(acs)-1], self.use_bias), g[l][n])
                a = self.activator.fn(z)
                if l == len(g)-1:
                    a = self.end_activator.fn(z)
                zs.append(z)
                acls.append(a)
            acs.append(acls)
            self.outputs.append(zs)
        return self.outputs, acs, acs[len(acs)-1]
    
    def backprop(self, y_c, inputs, lrn_rt):
        zs, acs, ys = self.predict(inputs)
        gradAC = []
        err = []
        dzs = []
        for i in range(len(ys)):
            dzs.append(self.end_activator.dfn(zs[len(zs)-1][i]))
        for i in range(len(y_c)):
            gradAC.append(ys[i]-y_c[i])
        gradAC = np.array(gradAC)
        dzs = np.array(dzs)
        err.append(dzs*gradAC)
    
        #calculate err of layers
        for layer in range(len(self.hidden)-2, -1, -1): #len(hidden)-2 because we already have the err for the output layer
            ws = []
            #transpose weights
            for neuron in range(len(self.hidden[layer+1])):
                ws.extend(self.hidden[layer+1][neuron][:-1])
            #for each layer: [
            #   for each neuron: [
            #       err
            #   ]
            # ]

            ws = np.array(ws)
            dzs = []
            for i in range(len(zs[layer+1])):
                dzs.append(self.activator.dfn(zs[layer+1][i]))
            print(ws)
            err.insert(0, np.multiply(np.multiply(np.transpose(ws), err[0]), dzs)) #get err of this layer in terms of next layer
        #use the err array to start correcting
        for layer in range(len(err)-1, -1, -1):
            for neuron in range(len(err[layer])):
                for weight in range(len(self.hidden[layer][neuron])):
                    self.hidden[layer][neuron][weight] -= lrn_rt*acs[layer-1][neuron]*err[layer][neuron]
                self.hidden[layer][neuron][len(self.hidden[layer][neuron])-1] -= self.use_bias*lrn_rt*err[layer][neuron]
        return self.hidden
    def import_from_file(self, path):
        #old node format:[[weights], bias]
        #new node format:[weights, bias]
        with open(path, 'r') as o:
            d = json.load(o)
            self.hidden = np.array(d['hidden'])
            self.inputs = np.array(d['inputs'])

        return self
    
    def into_list(self):
        ins = []
        for i in self.inputs:
            ins.append(float(str(i)))
        h = []
        for l in self.hidden:
            ls = []
            for n in l:
                ns = []
                for k in n:
                    ns.append(float(str(k)))
                ls.append(ns)
            h.append(ls)
        return ins, h


    def output_to_file(self, path):
        ls = self.into_list()
        with open(path, 'w') as f:
            json.dump({'inputs':ls[0], 'hidden':ls[1]}, f)
        return

#num_correct = 0
#num_wrong = 0
#
#corr_t = []
#c_t = 0
#iters = 1000
#initializer = tf.keras.initializers.HeUniform()
#model = tf.keras.models.Sequential([
#    tf.keras.layers.Input(shape=(2,)),
#    tf.keras.layers.Dense(4, activation='relu', kernel_initializer=initializer),
#    tf.keras.layers.Dense(1, activation='relu', kernel_initializer=initializer)
#
#])
#model.compile(optimizer="SGD", loss="mae")
#for o in range(1, iters+1):
#    _1 = randint(0, 5)
#    _2 = randint(0, 5)
#    x = np.array([[_1, _2]])
#    out = model(x)
#    y = out[0][0]
#    correct = _1+_2
#    if int(y*10+0.5) != correct:
#        num_wrong += 1
#        model.fit(x, np.array([[(correct)/10]]))
#    else:
#        num_correct += 1
#        c_t += 1
#    print(o)
#    if o % (iters/10) == 0:
#        corr_t.append(10*c_t/iters)
#        c_t = 0
#        print("\n" + str(num_correct) + "-" + str(num_wrong) + ", " + str(o))
#
#print(num_correct/iters)
#print(num_wrong/iters)
#print(corr_t)