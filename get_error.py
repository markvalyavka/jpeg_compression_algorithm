import cv2
import numpy as np
import math


def get_error(A, A_hat):
    B = A - A_hat
    return math.sqrt(np.dot(B.transpose(), B).trace())
