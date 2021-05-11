import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import os
import cv2
from pympler.asizeof import asizeof
import time

import svd
from sys import getsizeof


def _power_iteration(B, q):
    for i in range(q):
        B = B @ (B.T @ B)
    return B


def rSVD(A, r, q, p):
    # Smaple columnspace of A matrix with P matrix
    new_n = A.shape[1]
    P = np.random.randn(new_n, r + p)
    B = A @ P

    B = _power_iteration(B, q)

    Q, _ = np.linalg.qr(B, mode='reduced')

    # Compute svd on projected Y
    Y = Q.T @ A
    Uy, S, Vt = np.linalg.svd(Y)
    U = Q @ Uy

    return U, S, Vt


def get_error(A, B):
    # return (np.square(A - B)).mean()
    return np.sqrt(np.sum(np.square(A - B)))
    # return np.linalg.norm(A - B, ord=2) / np.linalg.norm(A, ord=2)


if __name__ == '__main__':
    r = 200
    q = 1
    p = 5

    A = cv2.imread("test_files/mona_liza.jpg")
    A = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    A = np.array(A, dtype=np.float64)

    t = time.time()
    U, S, Vt = rSVD(A, r, q, p)
    # A_new = svd.recreate_approx_image(S, U, Vt.T, r)
    A_new = U[:, :(r + 1)] @ np.diag(S[:(r + 1)]) @ Vt[:(r + 1), :]

    # plt.imshow(A_new, cmap='gray')
    # plt.imshow(A_new - A)
    # plt.show()

    cv2.imwrite('./output_rand/original.png', A)
    cv2.imwrite(f'./output_rand/compressed{r}.png', A_new)

    print("Error: ", get_error(A, A_new))

    end_t = time.time()
    print("Time: ", end_t - t)
