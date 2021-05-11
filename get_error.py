import cv2
import numpy as np
import math


def get_error(A, B):
    return np.sqrt(np.sum(np.square(A - B)))
