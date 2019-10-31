#! /usr/bin/python3

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


def get_blank_matrix():
    return [[0 for _ in range(SIDE_PX)] for _ in range(SIDE_PX)]


def get_square_img():
    img = get_image()
    pixels = img.load()
    instance = get_blank_matrix()
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


def get_circle_img():
    img = get_image()
    draw = ImageDraw.Draw(img)
    draw.ellipse((1, 1, SIDE_PX - 2, SIDE_PX - 2), fill='white', outline='black')
    return img


def get_triangle_img():
    img = get_image()
    pixels = img.load()
    instance = get_blank_matrix()
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


def get_random_figure():
    figure = randint(0, 2)
    if figure == 0:
        img = get_square_img()
    elif figure == 1:
        img = get_circle_img()
    else:
        img = get_triangle_img()
    return img


def mutate_vector(vector_given):
    vec = [0 for _ in range(len(vector_given))]
    # describe mutating rules here
    for i in range(len(vector_given)):
        if vector_given[i]:
            vec[i] = 1
    return vec


def sygm(x):
    return 1 / (1 + np.exp(-x))


def is_white(x):
    return 1 if x > 100 else 0


def harass_img(img, depth):
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


def get_vector(img):
    bit_mask = img.load()
    vec = get_blank_matrix()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if not is_white(bit_mask[i, j]):
                vec[j][i] = 1
    return sum(vec, [])


def read_line(filename, delete=1):
    f = open(filename, "r")
    lines = f.readlines()
    f.close()
    string = lines[0]
    del lines[0]
    if delete:
        f = open(filename, "w")
        f.writelines(lines)
        f.close()
    return string


def str_to_vector(string):
    vec = [0 for _ in range(len(string) - 1)]
    for i in range(len(string)):
        if string[i] == '1':
            vec[i] = 1
    return vec


def generate_set(harass, size=100):
    f = open(GEN_SET_FILE, "w")
    for i in range(size):
        figure = randint(0, 2)
        figure_description = [0 for _ in range(OUT_NEURONS_NUM)]
        if figure == 0:
            img = get_square_img()
            figure_description[0] = 1
        elif figure == 1:
            img = get_circle_img()
            figure_description[1] = 1
        else:
            img = get_triangle_img()
            figure_description[2] = 1
        print(''.join(str(e) for e in get_vector(harass_img(img, harass))), file=f)
        print(''.join(str(e) for e in figure_description), file=f)
    f.close()


def run_set(net):
    net.show_weights()
    for i in range(SET_SIZE):
        pair_from_base_file = str_to_vector(read_line(GEN_SET_FILE)), str_to_vector(read_line(GEN_SET_FILE))
        net.eat(pair_from_base_file[0])
        result = net.run()
        max_val = max(pair_from_base_file[1])
        waited = 0
        for i in range(len(pair_from_base_file[1])):
            if pair_from_base_file[1][i] == max_val:
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
            mutate_weights(net, pair_from_base_file[1])
    net.show_weights()
    net.show_delta_weights()


def mutate_weights(net, need):
    for layer_num in range(len(net.layers) - 1, -1, -1):
        if layer_num == len(net.layers) - 1:
            i = 0
            for neuron in net.layers[layer_num].neurons:
                _out = net.vectors[layer_num + 1][i]
                t = need[i]
                neuron.delta = (t - _out) * (1 - _out) * _out
                _in = net.vectors[layer_num]
                for pr_nrn in range(len(net.layers[layer_num - 1].neurons)):
                    neuron.delta_weights[pr_nrn] = ALPHA_COEFFICIENT * neuron.delta * _in[pr_nrn]
                i += 1
        else:
            i = 0
            for neuron in net.layers[layer_num].neurons:
                delta_sum = 0
                _out = net.vectors[layer_num + 1][i]
                for nxt_nrn in net.layers[layer_num + 1].neurons:
                    delta_sum += nxt_nrn.delta * nxt_nrn.weights[i]
                neuron.delta = delta_sum * (1 - _out) * _out
                _in = net.vectors[layer_num]
                if layer_num == 0:
                    _range = len(net.vectors[0])
                else:
                    _range = len(net.layers[layer_num - 1].neurons)
                for pr_nrn in range(_range):
                    neuron.delta_weights[pr_nrn] = ALPHA_COEFFICIENT * neuron.delta * _in[pr_nrn]
                i += 1
    net.update()


class Network:
    def __init__(self, config, setup_vector):
        self.recognized = 0
        self.not_recognized = 0
        self.input_vector = setup_vector
        self.config = config
        self.layers = [0 for _ in range(len(self.config))]
        for i in range(0, len(self.layers)):
            if not i:
                self.layers[i] = Layer(self.config[i], len(self.input_vector))
            else:
                self.layers[i] = Layer(self.config[i], self.config[i - 1])
        self.vectors = [0 for _ in range(len(self.config) + 1)]
        for i in range(0, len(self.vectors)):
            if not i:
                self.vectors[i] = self.input_vector
            else:
                self.vectors[i] = [0 for _ in range(self.config[i - 1])]

    def run(self):
        for i in range(len(self.layers)):
            self.vectors[i + 1] = self.layers[i].run(self.vectors[i])
        return self.vectors[i + 1]

    def show_weights(self, layer=0):
        if not layer:
            for i in range(len(self.layers)):
                self.layers[i].show_weights()
        else:
            self.layers[layer].show_weights()

    def show_delta_weights(self, layer=0):
        if not layer:
            for i in range(len(self.layers)):
                self.layers[i].show_delta_weights()
        else:
            self.layers[layer].show_delta_weights()

    def show_delta(self, layer=0):
        if not layer:
            for i in range(len(self.layers)):
                self.layers[i].show_delta()
        else:
            self.layers[layer].show_weights()

    def update(self):
        for i in range(len(self.layers)):
            self.layers[i].update()

    def eat(self, vector):
        self.vectors[0] = vector


class Layer(Network):
    def __init__(self, neurons_num, vector_size):
        self.neurons = [0 for _ in range(neurons_num)]
        for i in range(len(self.neurons)):
            self.neurons[i] = Neuron(vector_size, sygm)

    def run(self, vector):
        vec = [0 for _ in range(len(self.neurons))]
        for i in range(len(self.neurons)):
            vec[i] = self.neurons[i].run(vector)
        return vec

    def show_weights(self):
        for i in range(len(self.neurons)):
            self.neurons[i].show_weights()

    def show_delta_weights(self):
        for i in range(len(self.neurons)):
            self.neurons[i].show_delta_weights()

    def show_delta(self):
        for i in range(len(self.neurons)):
            self.neurons[i].show_delta()

    def update(self):
        for i in range(len(self.neurons)):
            self.neurons[i].update()


class Neuron(Layer):
    def __init__(self, vector, func):
        self.func = func
        self.weights = [randint(-5, 5) / 10 for _ in range(vector)]
        self.delta_weights = [0.0 for _ in range(vector)]
        self.delta = 0.0

    def run(self, vector):
        res = 0
        for i in range(len(self.weights)):
            res += self.weights[i] * vector[i]
        return self.func(res)

    def show_weights(self):
        print(self.weights)

    def show_delta_weights(self):
        print(self.delta_weights)

    def update(self):
        for i in range(len(self.delta_weights)):
            self.weights[i] += self.delta_weights[i]


def main():
    if not (SIDE_PX % 2):
        print("bad side")
    else:
        harass_img(get_random_figure(), HARASS_LEVEL).save('img.png')
        generate_set(HARASS_LEVEL, SET_SIZE)
        net = Network(LAYERS_NUM_MAP, get_vector(get_random_figure()))
        run_set(net)
        print("recognized = ", net.recognized)
        print("not_recognized = ", net.not_recognized)


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
