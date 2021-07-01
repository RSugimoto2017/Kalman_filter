# -*- coding:UTF-8 -*-
import numpy


# Z~(k+1|k)
def calculate_Ztildekp1(zkp1, akp1, xhat):
    temp = numpy.dot(akp1, xhat)
    Ztildekp1 = zkp1 - temp
    return Ztildekp1


# S(k+1)
def calculate_Skp1(akp1, Pk, Rkp1):
    Skp1 = numpy.dot(akp1, Pk)
    Skp1 = numpy.dot(Skp1, akp1.T)
    Skp1 = Skp1 + Rkp1
    return Skp1


# W(k+1)
def calculate_Wkp1(Pk, akp1, Skp1):
    Wkp1 = numpy.dot(Pk, akp1.T)
    Wkp1 = numpy.dot(Wkp1, numpy.linalg.inv(Skp1))
    return Wkp1


# P(k|k)
def calculate_Pkp1(Pk, Wkp1, Skp1):
    temp = numpy.dot(Wkp1, Skp1)
    temp = numpy.dot(temp, Wkp1.T)
    Pkp1 = Pk - temp
    return Pkp1


# P(k+1|k)
def calculate_Pkp2(Pk, Q):
    Pkp2 = Pk + Q
    return Pkp2


# x^(k+1|k+1)
def calculate_xhatkp1(xhat, Wkp1, Ztildekp1):
    temp = numpy.dot(Wkp1, Ztildekp1)
    xhatkp1 = xhat + temp
    return xhatkp1


# 観測値行列z(k+1)
def get_zkp1(Z, k):
    zkp1 = [Z[(k+1)+15]]
    return numpy.array(zkp1)


# 行列a(k+1)
def get_akp1(A, k):
    akp1 = [A[(k+1)+15]]
    return numpy.array(akp1)


# 雑音の共分散行列R(k+1)
def get_Rkp1(R, k):
    Rkp1 = [R[(k+1)+15][(k+1)+15]]
    return numpy.array(Rkp1)


# カルマンフィルタ
def KalmanFiltering(xhat_init, Pk_init, Z, A, R):
    # init -> k=-16のときの初期値
    xhat = xhat_init
    Pk = Pk_init
    Q = Q_init

    print('推定値誤差の初期共分散P =\n', Pk)
    print('プラント雑音の共分散Q =\n', Q)

    for k in range(-16, 15):
        zkp1 = get_zkp1(Z, k)
        akp1 = get_akp1(A, k)
        Rkp1 = get_Rkp1(R, k)

        Ztildekp1 = calculate_Ztildekp1(zkp1, akp1, xhat)
        Skp1 = calculate_Skp1(akp1, Pk, Rkp1)
        Wkp1 = calculate_Wkp1(Pk, akp1, Skp1)
        Pk = calculate_Pkp1(Pk, Wkp1, Skp1)
        # print('推定値誤差の共分散P(', k+1, ') =\n', Pk)
        Pk = calculate_Pkp2(Pk, Q)
        # print('予測誤差の共分散P(', k+1, '|', k, ') =\n', Pk)
        xhat = calculate_xhatkp1(xhat, Wkp1, Ztildekp1)

        print('推定値x^(', k+1, ') =', xhat)
        # print(xhat[0])
        # print(xhat[1])
        # print(xhat[2])


# メイン
if __name__ == '__main__':

    # 必要な行列の用意
    # 観測回数k
    k = numpy.arange(-15, 16)

    # 観測値Z
    Z = numpy.array([162.1746, 139.5805, 113.8133, 94.3372, 74.7258, 59.3817, 41.4117, 26.5951, 20.1832, 8.8816, 1.8636, -5.0213, -5.8861, -5.7711, -4.9332, -
                     1.9845, 2.0593, 12.3849, 17.9044, 30.1826, 41.1677, 55.7128, 74.2944, 93.7607, 112.6638, 134.9818, 162.7143, 188.9610, 219.6236, 248.9036, 281.3082])

    # 係数行列A
    A = numpy.zeros((31, 3))
    for i in range(31):
        for j in range(3):
            if j == 0:
                A[i][j] = 1
            elif j == 1:
                A[i][j] = k[i]
            else:
                A[i][j] = k[i]*k[i]

    # 観測雑音wの共分散行列R(31*31)
    R = numpy.zeros((31, 31))
    for i in range(31):
        for j in range(31):
            # 対角成分
            if j == i:
                # kが奇数の場合のセンサノイズ
                if k[i] % 2 != 0:
                    R[i][j] = 1.0
                # kが偶数の場合センサノイズ
                else:
                    R[i][j] = 4.0
            # 非対角成分
            else:
                R[i][j] = 0

    # 推定値xhatの初期値
    xhat_init = numpy.zeros(3)
    # 推定値誤差の初期共分散行列P
    Pk_init = numpy.zeros((3, 3))
    # プラント雑音の共分散行列Q
    Q_init = numpy.zeros((3, 3))

    # 対角成分の初期値を設定(推定値誤差の初期共分散)
    numpy.fill_diagonal(Pk_init, 1000000)
    # 対角成分の初期値を設定(プラント雑音の共分散)
    numpy.fill_diagonal(Q_init, 0.001)

    print('Kalman Filtering')
    KalmanFiltering(xhat_init, Pk_init, Z, A, R)
