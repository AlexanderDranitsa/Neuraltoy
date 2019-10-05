from random import randint
import sys

import numpy as np
from PIL import Image, ImageDraw

WHITE_CODE = 255
BLACK_CODE = 0
OUT_NEURONS_NUM = 3
GEN_SET_FILE = "GeneratedSet.txt"
ALPHA_COEFFICIENT = 15

def get_image():
    return Image.new('L', (SIDE_PX, SIDE_PX), color='white')

def Blank():
    return [[0 for _ in range(SIDE_PX)] for _ in range(SIDE_PX)]


def Msquare():
    img = get_image()
    pixels = img.load()
    instance = Blank()
    border = SIDE_PX - 1
    for i in range(1, border):
        instance[1][i] = 1
        instance[border - 1][i] = 1
        instance[i][1] = 1
        instance[i][border - 1] = 1
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                if instance[i][j]:
                    pixels[i, j] = BLACK_CODE
    return img


def Mcircle():
    img = get_image()
    draw = ImageDraw.Draw(img)
    draw.ellipse((1, 1, SIDE_PX - 2, SIDE_PX - 2), fill='white', outline='black')
    return img


def Mtriangle():
    img = get_image()
    pixels = img.load()
    instance = Blank()
    border = SIDE_PX - 1
    cap = SIDE_PX // 2
    i = 1
    while i <= cap:
        instance[i][border - i] = 1
        instance[border - i][SIDE_PX - i - 1] = 1
        i = i + 1
    for i in range(1, border):
        instance[i][border - 1] = 1
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if instance[i][j]:
                pixels[i, j] = BLACK_CODE
    return img


def Randfigure():
    figure = randint(0, 2)
    if figure == 0:
        img = Msquare()
    elif figure == 1:
        img = Mcircle()
    else:
        img = Mtriangle()
    return img


def MutateVector(Vector):
    newVector = [0 for _ in range(len(Vector))]
    # describe mutating rules here
    for i in range(len(Vector)):
        if Vector[i]:
            newVector[i] = 1
    return newVector


def Sygm(x):
    return 1 / (1 + np.exp(-x))


def is_white(x):
    return 1 if x > 100 else 0


def Harass(img, depth):
    while depth:
        pixels = img.load()
        x = randint(0, SIDE_PX - 1)
        y = randint(0, SIDE_PX - 1)
        if not is_white(pixels[x, y]):
            pixels[x, y] = WHITE_CODE
            not_added = 1
            while not_added:
                i = randint(0, SIDE_PX - 1)
                j = randint(0, SIDE_PX - 1)
                if is_white(pixels[i, j]):
                    pixels[i, j] = BLACK_CODE
                    not_added = 0
                    depth -= 1
    return img


def GetVec(img):
    bitmask = img.load()
    Vector = Blank()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if not is_white(bitmask[i, j]):
                Vector[j][i] = 1
    return sum(Vector, [])


def ToImg(Vector):
    img = get_image()
    pixels = img.load()
    for i in range(SIDE_PX * SIDE_PX):
        if Vector[i] == '1':
            pixels[(i % SIDE_PX), (i // SIDE_PX)] = 1
    return img


def ReadLn(fname, delete=1):
    f = open(fname, "r")
    lines = f.readlines()
    f.close()
    string = lines[0]
    del lines[0]
    if delete:
        f = open(fname, "w")
        f.writelines(lines)
        f.close()
    return string


def ToVector(string):
    Vector = [0 for _ in range(len(string) - 1)]
    for i in range(len(string)):
        if string[i] == '1':
            Vector[i] = 1
    return Vector


def ReadFromBase():
    Vector = ToVector(ReadLn(GEN_SET_FILE))
    Figure = ToVector(ReadLn(GEN_SET_FILE))
    return (Vector, Figure)


def GenerateSet(harass, size=100):
    f = open(GEN_SET_FILE, "w")
    for i in range(size):
        figure = randint(0, 2)
        Res = [0 for _ in range(OUT_NEURONS_NUM)]
        if figure == 0:
            img = Msquare()
            Res[0] = 1
        elif figure == 1:
            img = Mcircle()
            Res[1] = 1
        else:
            img = Mtriangle()
            Res[2] = 1
        Vector = GetVec(Harass(img, harass))
        VectorStr = ''.join(str(e) for e in Vector)
        ResStr = ''.join(str(e) for e in Res)
        print(VectorStr, file=f)
        print(ResStr, file=f)
    f.close()


def CalculateErr(need, got):
    TotalErr = 0
    for i in range(len(need)):
        TotalErr += need[i] - got[i]
    return TotalErr * 0.5


def RunSet(net):
    net.ShowWeights()
    for i in range(SET_SIZE):
        couple = ReadFromBase()
        net.Eat(couple[0])
        result = net.Run()
        max_val = max(couple[1])
        waited = 0
        for i in range(len(couple[1])):
            if couple[1][i] == max_val:
                waited = i
        got = 0
        max_val = max(result)
        for i in range(len(result)):
            if result[i] == max_val:
                got = i
        if waited == got:
            net.recognized += 1
        else:
            net.not_recognized += 1
            MutateWeights(net, couple[1])
    net.ShowWeights()
    net.ShowdWeights()


def MutateWeights(net, need):
    for layer_num in range(len(net.Layers) - 1, -1, -1):
        if layer_num == len(net.Layers) - 1:
            i = 0
            for neuron in net.Layers[layer_num].Neurons:
                _out = net.Vectors[layer_num + 1][i]
                t = need[i]
                neuron.Delta = (t - _out) * (1 - _out) * _out
                _in = net.Vectors[layer_num]
                for pr_nrn in range(len(net.Layers[layer_num - 1].Neurons)):
                    neuron.dWeights[pr_nrn] = ALPHA_COEFFICIENT * neuron.Delta * _in[pr_nrn]
                i += 1
        else:
            i = 0
            for neuron in net.Layers[layer_num].Neurons:
                delta_sum = 0
                _out = net.Vectors[layer_num + 1][i]
                for nxt_nrn in net.Layers[layer_num + 1].Neurons:
                    delta_sum += nxt_nrn.Delta * nxt_nrn.Weights[i]
                neuron.Delta = delta_sum * (1 - _out) * _out
                _in = net.Vectors[layer_num]
                if layer_num == 0:
                    _range = len(net.Vectors[0])
                else:
                    _range = len(net.Layers[layer_num - 1].Neurons)
                for pr_nrn in range(_range):
                    neuron.dWeights[pr_nrn] = ALPHA_COEFFICIENT * neuron.Delta * _in[pr_nrn]
                i += 1
    net.Update()


class Network:
    def __init__(self, config, setupvector):
        self.recognized = 0
        self.not_recognized = 0
        self.InputVector = setupvector
        self.Config = config
        self.Layers = [0 for _ in range(len(self.Config))]
        for i in range(0, len(self.Layers)):
            if not i:
                self.Layers[i] = Layer(self.Config[i], len(self.InputVector))
            else:
                self.Layers[i] = Layer(self.Config[i], self.Config[i - 1])
        self.Vectors = [0 for _ in range(len(self.Config) + 1)]
        for i in range(0, len(self.Vectors)):
            if not i:
                self.Vectors[i] = self.InputVector
            else:
                self.Vectors[i] = [0 for _ in range(self.Config[i - 1])]

    def Run(self):
        for i in range(len(self.Layers)):
            self.Vectors[i + 1] = self.Layers[i].Run(self.Vectors[i])
        return self.Vectors[i + 1]

    def ShowVectors(self, num='EMPTY'):
        if num == 'EMPTY':
            for i in range(len(self.Vectors)):
                print(self.Vectors[i])
        else:
            self.Vectors[num].ShowWeights()
        return self.Vectors

    def ShowWeights(self, layer=0):
        if not layer:
            for i in range(len(self.Layers)):
                self.Layers[i].ShowWeights()
        else:
            self.Layers[layer].ShowWeights()

    def ShowdWeights(self, layer=0):
        if not layer:
            for i in range(len(self.Layers)):
                self.Layers[i].ShowdWeights()
        else:
            self.Layers[layer].ShowdWeights()

    def ShowDelta(self, layer=0):
        if not layer:
            for i in range(len(self.Layers)):
                self.Layers[i].ShowDelta()
        else:
            self.Layers[layer].ShowWeights()

    def Update(self):
        for i in range(len(self.Layers)):
            # print("layer %d update" %i)
            self.Layers[i].Update()

    def Eat(self, vector):
        self.Vectors[0] = vector


class Layer(Network):
    def __init__(self, neurons_num, vectorsize):
        self.Neurons = [0 for _ in range(neurons_num)]
        for i in range(len(self.Neurons)):
            self.Neurons[i] = Neuron(vectorsize, Sygm)

    def Run(self, vector):
        Vector = [0 for _ in range(len(self.Neurons))]
        for i in range(len(self.Neurons)):
            Vector[i] = self.Neurons[i].Run(vector)
        return Vector

    def ShowWeights(self):
        for i in range(len(self.Neurons)):
            self.Neurons[i].ShowWeights()

    def ShowdWeights(self):
        for i in range(len(self.Neurons)):
            self.Neurons[i].ShowdWeights()

    def ShowDelta(self):
        for i in range(len(self.Neurons)):
            self.Neurons[i].ShowDelta()

    def Update(self):
        for i in range(len(self.Neurons)):
            # print(">neuron %d update" %i)
            self.Neurons[i].Update()


class Neuron(Layer):
    def __init__(self, vector, func):
        self.Func = func
        self.Weights = [randint(-5, 5) / 10 for _ in range(vector)]
        self.dWeights = [0.0 for _ in range(vector)]
        self.Delta = 0.0

    def Run(self, vector):
        res = 0
        for i in range(len(self.Weights)):
            res += self.Weights[i] * vector[i]
        return self.Func(res)

    def ShowWeights(self):
        print(self.Weights)

    def ShowdWeights(self):
        print(self.dWeights)

    def Update(self):
        for i in range(len(self.dWeights)):
            self.Weights[i] += self.dWeights[i]


def main():
    if not (SIDE_PX % 2):
        print("bad side")
    else:
        img = Randfigure()
        Vector = MutateVector(GetVec(img))
        img = Harass(img, HARASS_LEVEL)
        img.save('img.png')

        Vector = GetVec(img)
        GenerateSet(HARASS_LEVEL, SET_SIZE)
        Net = Network(LAYERS_NUM_MAP, Vector)
        RunSet(Net)
        print("recognized = ", Net.recognized)
        print("not_recognized = ", Net.not_recognized)


if __name__ == '__main__':
    if not len(sys.argv) == 4:
        print("ARGS:\n"
              "1: generated set size\n"
              "2: img side size in px\n"
              "3: how hard img passed to trained net should be harassed (number of swapped px)")
    else:
        SET_SIZE = int(sys.argv[1])
        SIDE_PX = int(sys.argv[2])
        HARASS_LEVEL = int(sys.argv[3])
        LAYERS_NUM_MAP = [5, 4, OUT_NEURONS_NUM]
        main()
