import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pprint

from math import cos, sqrt, pi

img = cv.imread("mona_liza.jpeg")

# Split into 3 color channels
B, G, R = cv.split(img)

# Merge 3 channels back


img = cv.merge([R, G, B])
#plt.imshow(img)
#plt.show()





def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return np.uint8(ycbcr)


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float32)
    rgb[:, :, [1, 2]] -= 128
    return np.uint8(rgb.dot(xform.T))


def get_dctmtx(n):
    """
    Get dct n-dimensional coefficient matrix C

    :param n: dimension of dct coefficient matrix
    :return: dct coefficient matrix
    """
    def calculate_dctmtx_entry(u, v, n):
        if u == 0:
            return sqrt(1/n)
        else:
            return sqrt(2/n) * cos(((2*v+1)*pi*u)/(2*n))

    C = np.zeros((n, n))

    for u in range(n):
        for v in range(n):
            C[u, v] = round(calculate_dctmtx_entry(u, v, n), 2)
    return C


def divide_into_block(n, img_channel):
    """
    Divides 2D matrix into n x n blocks (np.arrays)

    :param n: divide into n x n slices
    :param img_channel: 2D matrix
    :return: list of lists of n x n blocks
    """
    sliced = np.split(img_channel, n, axis=0)
    blocks = [np.split(img_slice, n, axis=1) for img_slice in sliced]
    return blocks


def group_blocks_together(blocks):
    """
    Groups n x n blocks (np.arrays) back to 2D matrix

    :param blocks: list of lists of n x n blocks
    :return: 2D matrix of grouped blocks
    """

    img_stacked = np.block(blocks)
    return img_stacked





#print(get_dctmtx(8))

