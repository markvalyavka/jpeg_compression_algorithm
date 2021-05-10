import cv2 as cv
import numpy as np
import sys
# from pympler.asizeof import asizeof
import matplotlib.pyplot as plt
from dct import forward_dct, backwards_dct
from zig_zag import zig_zag, inverse_zigzag
from huffman_encoding import huffman_encode_block, huffman_decode_block
from dahuffman import HuffmanCodec
import sys
from flask import Flask, render_template, request, redirect, url_for
import os
import glob

UPLOAD_FOLDER = './render_files'
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(os.path.join(UPLOAD_FOLDER, uploaded_file.filename))
        file = glob.glob(f'{UPLOAD_FOLDER}/*')[0]
        main(file)
        return redirect(url_for('index'))


# # plt.imshow(img)
# # plt.show()

def rgb2ycbcr(im):
    """
    Converts the RGB image to the YCbCr color space which
    separates the illuminance and chrominance.
    """
    xform = np.array([[.299, .587, .114],
                      [-.1687, -.3313, .5],
                      [.5, -.4187, -.0813]])  # explain why?
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return np.int16(ycbcr)


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402],
                      [1, -0.34414, -.71414],
                      [1, 1.772, 0]])  # explain why?
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
    # global IMG_WIDTH
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


def main(image):
    img = cv.imread(image)
    global IMG_WIDTH
    global IMG_HEIGHT
    IMG_WIDTH = len(img[0])
    IMG_HEIGHT = len(img)

    # Split into 3 color channels
    B, G, R = cv.split(img)

    # Merge 3 channels back
    img = cv.merge([R, G, B])
    # Convert RGB to YCbCr
    ycbcr_img = rgb2ycbcr(img)

    # Split each of the channels
    Cr, Cb, Y = cv.split(ycbcr_img)

    # Holds Cr_t, Cb_t, Y_t transformed channels
    # ready for Huffman encoding
    transformed_channels = []
    block_rows_num, block_cols_num = 0, 0

    # -----------------------------------------------
    # 1. Tranform color channel from RGB to CrCbY
    # -----------------------------------------------
    for img_channel in [Cr, Cb, Y]:
        img_channel_blocks = divide_into_blocks(8, img_channel)

        # -----------------------------------------------
        # 2. Perform DCT and Quantize
        # -----------------------------------------------
        forward_dct(img_channel_blocks)
        block_rows_num = len(img_channel_blocks)
        block_cols_num = len(img_channel_blocks[0])

        img_channel_combined = group_blocks_together(img_channel_blocks)
        transformed_channels.append(img_channel_combined)

    encoded_data = []

    for tranformed_channel in transformed_channels:
        channel_encoded_data = [[c for c in range(block_cols_num)] for r in range(block_rows_num)]
        tranformed_channel_blocks = divide_into_blocks(8, tranformed_channel)
        for block_row in range(len(tranformed_channel_blocks)):
            for block_col in range(len(tranformed_channel_blocks[0])):
                block = tranformed_channel_blocks[block_row][block_col]

                # -----------------------------------------------
                # 3. Zigzag each of 8x8 blocks separately
                # -----------------------------------------------
                zigzagged_block = zig_zag(block)

                # -----------------------------------------------
                # 4. Encode each of 8x8 blocks separately
                # -----------------------------------------------
                encoded_block = huffman_encode_block(zigzagged_block)
                channel_encoded_data[block_row][block_col] = encoded_block
        encoded_data.append(list(channel_encoded_data))

    # -----------------------------------------
    # For final presentation [Compare size of INPUT_DATA and COMPRESSED_DATA]
    # print(encoded_data)
    # print(asizeof(np.asarray(img)))
    # print(asizeof(np.asarray(encoded_data)))
    # ------------------------------------------
    huffman_decoded_data = []

    # -----------------------------------------------
    # 5. Decode data by Reversing each of the steps
    # -----------------------------------------------
    for encoded_channel in encoded_data:
        decoded_channel = [[c for c in range(block_cols_num)] for r in range(block_rows_num)]
        for block_row in range(len(encoded_channel)):
            for block_col in range(len(encoded_channel[0])):
                block = encoded_channel[block_row][block_col]
                decoded_block = huffman_decode_block(block[0], block[1])
                unzigzagged_block = inverse_zigzag(decoded_block, 8, 8)
                decoded_channel[block_row][block_col] = unzigzagged_block
        huffman_decoded_data.append(group_blocks_together(decoded_channel))

    # for channel in range(len(huffman_decoded_data)):
    #     huffman_decoded_data[channel] = group_blocks_together(huffman_decoded_data[channel])

    inv_dct_decoded_channels = []

    for img_channel in huffman_decoded_data:
        img_channel_blocks = divide_into_blocks(8, img_channel)

        backwards_dct(img_channel_blocks)

        img_channel_combined = group_blocks_together(img_channel_blocks)
        inv_dct_decoded_channels.append(img_channel_combined)

    # Combined Y, Cb, Cr channels to form an image
    decoded_img_ycbcr = cv.merge(inv_dct_decoded_channels)

    # Convert channels to RGB
    decoded_img_rgb = ycbcr2rgb(decoded_img_ycbcr)
    decoded_img_rgb = decoded_img_rgb.astype(np.uint8)

    # -----------------------------------------
    # For final presentation
    # for i in range(len(img)):
    #     for j in range(len(img[0])):
    #         for k in range(len(img[0][0])):
    #             if img[i][j][k] != decoded_img_rgb[i][j][k]:
    #                 print("fdsfd", img[i][j][k], decoded_img_rgb[i][j][k])
    # ------------------------------------------

    # Show img
    plt.imsave("./output/image.jpeg", decoded_img_rgb)
    # plt.imshow(decoded_img_rgb)
    # plt.show()


if __name__ == '__main__':
    app.run(debug=True)
