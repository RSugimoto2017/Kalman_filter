# -*- coding:UTF-8 -*-
import numpy as np


# 観測予測誤差（修正値）：Z~(k+1)
def calculate_Ztildekp1(zkp1, akp1, xhat):
    calculate = np.dot(akp1, xhat)
    Ztildekp1 = zkp1 - calculate
    return Ztildekp1


# 観測予測誤差共分散：S(k+1)
def calculate_Skp1(akp1, Pk, Rkp1):
    Skp1 = np.dot(akp1, Pk)
    Skp1 = np.dot(Skp1, akp1.T)
    Skp1 = Skp1 + Rkp1
    return Skp1


# フィルタゲイン：W(k+1)
def calculate_Wkp1(Pk, akp1, Skp1):
    Wkp1 = np.dot(Pk, akp1.T)
    Wkp1 = np.dot(Wkp1, np.linalg.inv(Skp1))
    return Wkp1


# 推定値：x^(k+1)
def calculate_xhatkp1(xhat, Wkp1, Ztildekp1):
    temp = np.dot(Wkp1, Ztildekp1)
    xhatkp1 = xhat + temp
    return xhatkp1


# 推定誤差共分散：P(k+1)
def calculate_Pkp1(Pk, Wkp1, Skp1):
    WSW = np.dot(Wkp1, Skp1)
    WSW = np.dot(WSW, Wkp1.T)
    Pkp1 = Pk - WSW
    return Pkp1


# 観測値：z(k+1)
def set_zkp1(Z, k):
    zkp1 = [Z[(k+1)+15]]
    return np.array(zkp1)


# 行列:a(k+1)
def set_akp1(A, k):
    akp1 = [A[(k+1)+15]]
    return np.array(akp1)


# 雑音の共分散：R(k+1)
def set_Rkp1(R, k):
    Rkp1 = [R[(k+1)+15][(k+1)+15]]
    return np.array(Rkp1)


# カルマンフィルタ
def KalmanFiltering(xhat_init, Pk_init, Z, A, R):
    # 初期値の設定
    xhat = xhat_init
    Pk = Pk_init

    # 単位行列：F
    F = np.eye(3)

    # プラント雑音の共分散：Q
    Q = np.zeros((3, 3))
    np.fill_diagonal(Q, 1)

    xhat_all = []
    for k in range(-16, 15):

        xhat = np.dot(xhat, F)

        Pk = np.dot(F, Pk)
        Pk = np.dot(Pk, F.T)
        Pk = Pk - Q

        zkp1 = set_zkp1(Z, k)
        akp1 = set_akp1(A, k)
        Rkp1 = set_Rkp1(R, k)

        # Z~(k+1)の計算
        Ztildekp1 = calculate_Ztildekp1(zkp1, akp1, xhat)
        # print('Z~(', k+1, ') =', Ztildekp1)

        # S(k+1)の計算
        Skp1 = calculate_Skp1(akp1, Pk, Rkp1)
        # print('S(', k+1, ') =', Skp1)

        # W(k+1)の計算
        Wkp1 = calculate_Wkp1(Pk, akp1, Skp1)
        # print('W(', k+1, ') = \n', Wkp1, '\n')

        # 推定値:xhat
        xhat = calculate_xhatkp1(xhat, Wkp1, Ztildekp1)
        print('推定値 x^(', k+1, ') = \n', xhat, '\n')
        xhat_all.append(xhat)

        # 推定誤差共分散:Pk
        Pk = calculate_Pkp1(Pk, Wkp1, Skp1)
        #print('推定誤差共分散 P(', k+1, ') = \n', Pk)

    # csvファイルの作成
    np.savetxt('result.csv', xhat_all, fmt="%.4f", delimiter=',')


if __name__ == '__main__':

    # 時刻：k=-15~15
    k = np.arange(-15, 16)

    # センサノイズの共分散行列：R
    R = np.zeros((31, 31))
    for i in range(31):
        # kが偶数の時のセンサノイズ
        if k[i] % 2 != 0:
            R[i][i] = 1.0
        # kが奇数の時のセンサノイズ
        else:
            R[i][i] = 4.0

    # 観測値z時刻kがkで変化したときのそれぞれの値：z
    Z = np.array([162.1746, 139.5805, 113.8133, 94.3372, 74.7258, 59.3817, 41.4117, 26.5951, 20.1832, 8.8816, 1.8636, -5.0213, -5.8861, -5.7711, -4.9332,
                  -1.9845, 2.0593, 12.3849, 17.9044, 30.1826, 41.1677, 55.7128, 74.2944, 93.7607, 112.6638, 134.9818, 162.7143, 188.9610, 219.6236, 248.9036, 281.3082])

    # 行列A
    A = np.zeros((31, 3))
    for i in range(31):
        for j in range(3):
            if j == 0:
                A[i][j] = 1
            elif j == 1:
                A[i][j] = k[i]
            else:
                A[i][j] = k[i]*k[i]

    # xhatの初期値
    xhat_init = np.zeros(3)
    # Pの初期値
    Pk_init = np.zeros((3, 3))
    np.fill_diagonal(Pk_init, 1000000)

    print("KalmanFiltering!!")
    KalmanFiltering(xhat_init, Pk_init, Z, A, R)
