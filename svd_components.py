import cv2
import numpy as np
import math

from image_compression import A_float64


def get_u(v, sigma):
    u = []
    for i in range(len(v)):
        ui = (1 / sigma[i]) * np.dot(A_float64, v[:, i])
        u.append(ui)

    return np.transpose(np.array(u))


def get_v(u, sigma):
    v = []
    for i in range(len(u)):
        vi = (1 / sigma[i]) * np.dot(A_float64.transpose(), u[:, i])
        v.append(vi)

    return np.transpose(np.array(v))
