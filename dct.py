import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pprint

from math import cos, sqrt, pi

img = cv.imread("panda_grayscale.png")

IMG_WIDTH = len(img[0])
IMG_HEIGHT = len(img)

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
    return np.int16(ycbcr)


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float32)
    rgb[:, :, [1, 2]] -= 128
    return np.int16(rgb.dot(xform.T))


def get_quantizationmtx():
    # Quantization matrix
    Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])
    return Q


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


def normalize_img_channel(img_channel):
    """
    Normalizes img_channel (2D matrix) if it is not divisible by 8
    """
    # todo implement
    pass


def divide_into_blocks(n, img_channel):
    """
    Divides 2D matrix into n x n blocks (np.arrays)

    :param n: divide into n x n slices
    :param img_channel: 2D matrix
    :return: list of lists of n x n blocks
    """
    sliced = np.split(img_channel, IMG_HEIGHT // n, axis=0)
    blocks = [np.split(img_slice, IMG_WIDTH // n, axis=1) for img_slice in sliced]
    return blocks


def group_blocks_together(blocks):
    """
    Groups n x n blocks (np.arrays) back to 2D matrix

    :param blocks: list of lists of n x n blocks
    :return: 2D matrix of grouped blocks
    """

    img_stacked = np.block(blocks)
    return img_stacked


def forward_dct(blocks):

    DCT_T = get_dctmtx(8)
    Q = get_quantizationmtx()

    for block_row in range(len(blocks)):

        for block_col in range(len(blocks[0])):

            # Normalize values around 0 by subtracting 128 from each entry
            # Needed because DCT works with values in range [-128, 127]
            blocks[block_row][block_col] -= 128

            # Perform DCT by using matrix multiplication
            # D = T M T'
            blocks[block_row][block_col] = np.linalg.multi_dot([DCT_T, blocks[block_row][block_col], DCT_T.transpose()])

            # Quantize using quantization matrix Q
            blocks[block_row][block_col] = np.round(np.divide(blocks[block_row][block_col], Q)) + 0


def backwards_dct(blocks):

    DCT_T = get_dctmtx(8)
    Q = get_quantizationmtx()

    for block_row in range(len(blocks)):

        for block_col in range(len(blocks[0])):

            # De-quantization step (Where error happens)
            blocks[block_row][block_col] = np.multiply(blocks[block_row][block_col], Q)

            # Invert DCT
            blocks[block_row][block_col] = np.linalg.multi_dot([DCT_T.transpose(), blocks[block_row][block_col], DCT_T])

            # Add 128 back
            blocks[block_row][block_col] += 128


if __name__ == "__main__":

    # Convert RGB to YcBcR
    ycbcr_img = rgb2ycbcr(img)

    # Split each of the channels
    Cr, Cb, Y = cv.split(ycbcr_img)

    # Holds Cr_t, Cb_t, Y_t transformed channels
    # ready for Huffman encoding
    transformed_channels = []

    for img_channel in [Cr, Cb, Y]:

        img_channel_blocks = divide_into_blocks(8, img_channel)
        forward_dct(img_channel_blocks)

        img_channel_combined = group_blocks_together(img_channel_blocks)
        transformed_channels.append(img_channel_combined)



    decoded_channels = []

    for img_channel in transformed_channels:

        img_channel_blocks = divide_into_blocks(8, img_channel)
        backwards_dct(img_channel_blocks)

        img_channel_combined = group_blocks_together(img_channel_blocks)
        decoded_channels.append(img_channel_combined)


    # Combined Y, Cb, Cr channels to form an image
    decoded_img_ycbcr = cv.merge(decoded_channels)

    # Convert channels to RGB
    decoded_img_rgb = ycbcr2rgb(decoded_img_ycbcr)
    decoded_img_rgb = decoded_img_rgb.astype(np.uint8)

    # Show img
    plt.imsave("./lsd.jpg", decoded_img_rgb)
    plt.imshow(decoded_img_rgb)
    plt.show()


