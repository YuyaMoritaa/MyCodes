# -*- coding: utf-8 -*-
# 固有顔(Eigenface)を用いた顔認証

import os
import cv2
import numpy as np
from pylab import *

# 顔画像データベース
Database = 'att_faces'

# 被験者数
person = 40
# 学習用顔画像の枚数（1人あたり）
training_faces_count = 6
# テスト用顔画像の枚数（1人あたり）
test_faces_count = 4
# 学習用顔画像の全枚数
l = training_faces_count * person

# 学習用画像のid
training_ids = [1, 2, 3, 4, 5, 6]
# テスト用画像のid
test_ids = [7, 8, 9, 10]

# 画像サイズ（縦）
height = 112
# 画像サイズ（横）
width = 92
# 画素数
pixels = height * width

# 累積寄与率(cumulative contribution ratio)の閾値
ccr_th = 0.85

# 識別関数
flag = 0
def classify(path, mean_img_col, evectors, W):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_col = np.array(img, dtype='float64').flatten()
    img_col -= mean_img_col
    img_col = np.reshape(img_col, (pixels, 1))

    S = evectors.T * img_col
    diff = W - S
    norms = np.linalg.norm(diff, axis=0)

    global flag
    if flag==0:
        print("・テスト画像のサイズ：{}".format(img_col.shape))
        print("・固有空間に射影後のテスト画像のサイズ：{}".format(S.shape))
        flag = 1

    # 最も類似した顔画像のidを求める
    closest_face_id = np.argmin(norms)

    return (closest_face_id // training_faces_count) + 1

############################################################
print("学習モード：")

# 各列が1枚の顔画像に対応する行列Lを定義
L = np.empty(shape=(pixels, l), dtype='float64')

cur_img = 0
for face_id in range(1, person+1):
    for training_id in training_ids:
        path = Database + "/" + "s" + str(face_id).zfill(2) +  "/" + str(training_id) + ".pgm"
        # 顔画像をグレイスケール画像として読み込み
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # 2次元配列を1次元配列に変換
        img_col = np.array(img, dtype='float64').flatten()
        # 1次元配列をLに格納
        L[:, cur_img] = img_col[:]

        cur_img += 1

# Lより平均画像を求める
mean_img_col = np.sum(L, axis=1) / l
# Lの各画像から平均画像を引く
for j in range(0, l):
    L[:, j] -= mean_img_col[:]

# Lから共分散行列Cを求める
# 注：計算コスト削減のためL^{T}Lの固有値問題を解く
C = np.matrix(L.T) * np.matrix(L)
C /= l
print("・行列Cのサイズ：", C.shape)

# 固有値と固有ベクトルを求める（実対称行列用のeighを使用）
# evectorsの各列に固有ベクトルが格納される
evalues, evectors = np.linalg.eigh(C)
print("・固有値の数：", evalues.shape)
print("・固有ベクトルを格納した配列のサイズ", evectors.shape)

# 固有値と固有ベクトルを降順に並べ替える
sort_indices = evalues.argsort()[::-1]
evalues = evalues[sort_indices]
evectors = evectors[:,sort_indices]

# 累積寄与率が閾値を超えるのに必要な固有値の最小数を求める
evalues_sum = sum(evalues[:])
evalues_count = 0
ccr = 0.0
for evalue in evalues:
    evalues_count += 1
    ccr += evalue / evalues_sum
    if ccr > ccr_th:
        break

print("・累積寄与率{}を超えるのに必要な固有値の数：{}".format(ccr_th, evalues_count))
evalues = evalues[0:evalues_count]
evectors = evectors[:,0:evalues_count]

# 本来の共分散行列の固有ベクトルを求めるためLを乗ずる
evectors = L * evectors

norms = np.linalg.norm(evectors, axis=0)
evectors = evectors / norms

print("・使用する固有値の数：", evalues.shape)
print("・使用する固有ベクトルを格納した配列のサイズ：", evectors.shape)

# 重み係数を求める
W = evectors.T * L
print("・重み係数Wを格納した配列のサイズ：", W.shape)

np.savetxt("evalues.dat", evalues)
np.savetxt("evectors.dat", evectors)
np.savetxt("weights.dat", W)

###########################################################
print("\nテストモード(被験者と識別結果)：")

test_count = test_faces_count * person
test_correct = 0
for face_id in range(1, person+1):
    for test_id in test_ids:
        path = Database + "/" + "s" + str(face_id).zfill(2) +  "/" + str(test_id) + ".pgm"
        result_id = classify(path, mean_img_col, evectors, W)
        result = (result_id == face_id)
        
#       print(face_id, result_id, result)
        print("{:02} {:02} {}".format(face_id, result_id, result))
        if result == True:
            test_correct += 1

accuracy = float(100 * test_correct / test_count)
print("正答率：", end="")
print(str(test_correct) + "/" + str(test_count) + " = ", end="")
print(str(accuracy) + " %")

# 平均顔と固有顔の表示
figure()
gray()
subplot(2, 4, 1)
imshow(mean_img_col.reshape(height, width))
for i in range(7):
    subplot(2, 4, i+2)
    imshow(evectors.T[i].reshape(height, width))
show()
