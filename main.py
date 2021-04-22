import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from dct import forward_dct, backwards_dct

img = cv.imread("test_files/panda_grayscale.png")

IMG_WIDTH = len(img[0])
IMG_HEIGHT = len(img)

# Split into 3 color channels
B, G, R = cv.split(img)

# Merge 3 channels back
img = cv.merge([R, G, B])


# plt.imshow(img)
# plt.show()


def rgb2ycbcr(im):
    """
    Converts the RGB image to the YCbCr color space which
    separates the illuminance and chrominance.
    """
    xform = np.array([[.299, .587, .114],
                      [-.1687, -.3313, .5],
                      [.5, -.4187, -.0813]])    # explain why?
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return np.int16(ycbcr)


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402],
                      [1, -0.34414, -.71414],
                      [1, 1.772, 0]])           # explain why?
    rgb = im.astype(np.float32)
    rgb[:, :, [1, 2]] -= 128
    return np.int16(rgb.dot(xform.T))


def normalize_img_channel(image_channel):
    """
    Normalizes img_channel (2D matrix) if it is not divisible by 8
    """
    # todo implement
    pass


def divide_into_blocks(n, image_channel):
    """
    Divides 2D matrix into n x n blocks (np.arrays)

    :param n: divide into n x n slices
    :param image_channel: 2D matrix
    :return: list of lists of n x n blocks
    """
    sliced = np.split(image_channel, IMG_HEIGHT // n, axis=0)
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


if __name__ == "__main__":
    # Convert RGB to YCbCr
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



    # for img_channel in transformed_channels:
    #     img_channel_blocks = divide_into_blocks(8, img_channel)
    #     backwards_dct(img_channel_blocks)
    #
    #     img_channel_combined = group_blocks_together(img_channel_blocks)
    #     decoded_channels.append(img_channel_combined)
    #
    # # Combined Y, Cb, Cr channels to form an image
    # decoded_img_ycbcr = cv.merge(decoded_channels)
    #
    # # Convert channels to RGB
    # decoded_img_rgb = ycbcr2rgb(decoded_img_ycbcr)
    # decoded_img_rgb = decoded_img_rgb.astype(np.uint8)

    # Show img
    # plt.imsave("./lsd.jpg", decoded_img_rgb)
    # plt.imshow(decoded_img_rgb)
    # plt.show()
