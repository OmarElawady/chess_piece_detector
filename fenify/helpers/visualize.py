from matplotlib import pyplot as plt
import cv2
from math import sqrt, ceil

def visualize_samples(x_samples, y_samples, n=16):
    n = min(len(x_samples), n)
    dimension_length = ceil(sqrt(n))
    for i in range(n):
        plt.subplot(dimension_length, dimension_length, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        visualize_sample(x_samples[i], y_samples[i])
    plt.show()

def convert_int_to_piece(n):
    pieces = 'rbnkqpRBNKQP'
    if n == 12:
        return 'E'
    else:
        return pieces[n]

def visualize_sample(img, piece):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.xlabel(piece)
    plt.ylabel(convert_int_to_piece(piece))
