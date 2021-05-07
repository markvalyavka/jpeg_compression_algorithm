import collections
from dahuffman import HuffmanCodec


def huffman_encode_block(zigzagged_block):

    frequencies = collections.Counter(zigzagged_block)
    huffman_codec = HuffmanCodec.from_frequencies(frequencies)
    temp = []
    for k, v in frequencies.items():
        temp.extend([k, v])
    return huffman_codec.encode(zigzagged_block), dict(frequencies)


def huffman_decode_block(encoded_block, block_frequencies):

    huffman_codec = HuffmanCodec.from_frequencies(block_frequencies)
    decoded_block = huffman_codec.decode(encoded_block)
    return decoded_block
