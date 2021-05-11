import cv2
import numpy as np
import math
from sys import getsizeof
import time

from matplotlib import pyplot as plt

from svd_components import *
from get_error import *
from compare import *

# A = cv2.imread("test_files/mona_liza.jpg")
A = cv2.imread("test_files/mona_liza.jpg")
A = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
A_float64 = np.array(A, dtype=np.float64)

k = 200

m = np.shape(A_float64)[0]
n = np.shape(A_float64)[1]


def recreate_approx_image(sigma, u, v, k):
    B = np.zeros((m, n))
    A_hat = np.zeros((m, n), dtype=np.uint8)

    for i in range(k):
        B = np.add(B, sigma[i] * np.outer(u[:, i], v[:, i]))

    for i in range(len(B)):
        for j in range(len(B[0])):
            if B[i][j] > 255:
                A_hat[i][j] = 255
            elif B[i][j] < 0:
                A_hat[i][j] = 0
            else:
                A_hat[i][j] = B[i][j]
    return A_hat


def app():
    if m > n:
        AtA = np.dot(A_float64.transpose(), A_float64)
        sigma, v = np.linalg.eig(AtA)
        u = get_u(v, sigma)
    else:
        AAt = np.dot(A_float64, A_float64.transpose())
        sigma, u = np.linalg.eig(AAt)
        v = get_v(u, sigma)

    A_hat = recreate_approx_image(sigma, u, v, k)
    error = get_error(A_float64, A_hat)
    print("Error:", error)

    cv2.imwrite('./output/original.png', A)
    cv2.imwrite(f'./output/compressed{k}.png', A_hat)

    return error, sigma


if __name__ == '__main__':
    t = time.time()
    error, sig = app()
    end_t = time.time()
    print("Time: ", end_t - t)

    #########################################################
    # Singular Values

    # plt.semilogy(np.diag(sig))
    # plt.title('Singular Values')

    #########################################################
    # Cumulative Sum of The Singular Values
    #
    # plt.plot(np.cumsum(np.diag(sig) / np.sum(np.diag(sig))), color='red')
    # plt.title('Cumulative Sum of The Singular Values')




    # perc_strg = []
    # x_ticks = []
    # rank = np.linalg.matrix_rank(A)
    # for r in np.linspace(1, rank, 10):
    #     x_ticks.append(r)
    #     perc_strg.append(perc_storage(r, n, m))
    #
    # print(perc_strg)
    #
    # plt.plot(x_ticks, perc_strg)
    # plt.xlabel('rank')
    # plt.ylim('% storage_required')
    # plt.title('rank   v/s   percentage_storage_required')

    # pyplot.tight_layout()

    # plt.show()

    ############################################################
    # Error K plot

    # errors = []
    # ks = []
    # for i in range(1, 250, 10):
    #     k = i
    #     er, sig = app()
    #     ks.append(k)
    #     errors.append(er)
    #
    # plt.plot(ks, errors, color='green')
    plt.show()