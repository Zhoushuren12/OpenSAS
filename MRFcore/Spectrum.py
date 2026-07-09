import numpy as np
import matplotlib.pyplot as plt
import time as t
import math
"""
计算反应谱
（Nigam-Jennings精确解）
"""


def Spectrum(ag, dt, T, zeta=0.05):

    ag = np.asarray(ag)
    T = np.asarray(T)
    if T[0] == 0:
        T = T[1:]
        mark = 1
    else:
        mark = 0
    N = len(T)
    if N == 1:
        w = 2 * math.pi / float(T[0])
        wd = w * math.sqrt(1 - zeta**2)
        u = 0.0
        v = 0.0
        RSA = 0.0
        RSV = 0.0
        RSD = 0.0
        B1 = math.exp(-zeta * w * dt) * math.cos(wd * dt)
        B2 = math.exp(-zeta * w * dt) * math.sin(wd * dt)
        w_2 = 1.0 / w ** 2
        w_3 = 1.0 / w ** 3
        for i in range(len(ag) - 1):
            p_i = -float(ag[i])
            alpha_i = (-float(ag[i + 1]) + float(ag[i])) / dt
            A0 = p_i * w_2 - 2.0 * zeta * alpha_i * w_3
            A1 = alpha_i * w_2
            A2 = u - A0
            A3 = (v + zeta * w * A2 - A1) / wd
            u = A0 + A1 * dt + A2 * B1 + A3 * B2
            v = A1 + (wd * A3 - zeta * w * A2) * B1 - (wd * A2 + zeta * w * A3) * B2
            a = -2 * zeta * w * v - w * w * u
            RSA = max(RSA, abs(a))
            RSV = max(RSV, abs(v))
            RSD = max(RSD, abs(u))
        RSA = np.array([RSA])
        RSV = np.array([RSV])
        RSD = np.array([RSD])
        if mark == 1:
            RSA = np.insert(RSA, 0, np.max(abs(ag)))
            RSV = np.insert(RSV, 0, 0)
            RSD = np.insert(RSD, 0, 0)
        return RSA, RSV, RSD

    w = 2 * np.pi / T
    wd = w * np.sqrt(1 - zeta**2)
    n = len(ag)

    u = np.zeros(N)
    v = np.zeros(N)
    RSA = np.zeros(N)
    RSV = np.zeros(N)
    RSD = np.zeros(N)

    B1 = np.exp(-zeta * w * dt) * np.cos(wd * dt)  # N
    B2 = np.exp(-zeta * w * dt) * np.sin(wd * dt)  # N

    w_2 = 1.0 / w ** 2  # N
    w_3 = 1.0 / w ** 3  # N

    for i in range(n - 1):
        p_i = -ag[i]
        alpha_i = (-ag[i + 1] + ag[i]) / dt

        A0 = p_i * w_2 - 2.0 * zeta * alpha_i * w_3  # N
        A1 = alpha_i * w_2  # N
        A2 = u - A0  # N
        A3 = (v + zeta * w * A2 - A1) / wd  # N

        u = A0 + A1 * dt + A2 * B1 + A3 * B2  # N
        v = A1 + (wd * A3 - zeta * w * A2) * B1 - (wd * A2 + zeta * w * A3) * B2  # N
        a = -2 * zeta * w * v - w * w * u  # N
        RSA = np.maximum(RSA, np.abs(a))
        RSV = np.maximum(RSV, np.abs(v))
        RSD = np.maximum(RSD, np.abs(u))
    if mark == 1:
        RSA = np.insert(RSA, 0, np.max(abs(ag)))
        RSV = np.insert(RSV, 0, 0)
        RSD = np.insert(RSD, 0, 0)
    return RSA, RSV, RSD

if __name__ == "__main__":

    ag = np.loadtxt(r'D:\Study\My_Project\OpenSAS-SMAPFDB\GMs\1.txt')
    dt = 0.02
    T1 = 0
    T2 = 6
    dT = 0.01
    save = 0  # 1：保存结果，2：不保存

    T = np.arange(T1, T2, dT)
    RSA, RSV, RSD = Spectrum(ag, dt, T)
    plt.plot(T, RSA)
    plt.show()
    if save == 1:
        np.savetxt('T.out', T)
        np.savetxt('RSA.out', RSA)
