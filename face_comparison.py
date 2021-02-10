# 顔画像の画素値比較による顔認証（最小二乗）

import os
import cv2
import numpy as np

# 顔画像データベース
ImgDatabase = 'att_faces'

# 被験者数
Npsn = 40

# 一被験者あたりの顔画像の枚数
Ndat = 10

# 画像間の距離を算出する関数
def distance(img1, img2):
  height = img1.shape[0]
  width  = img1.shape[1]
  err = np.sum((img1 - img2) ** 2)
  err /= (height * width)
  return err

# 本人間の距離の計算結果を保存する配列
gdist = np.array([])

# 本人・他人間の距離の計算結果を保存する配列
odist = np.array([])

# 被験者リスト（"s01", "s02", ...）
plist = ["s" + str(x).zfill(2) for x in range(1,Npsn+1)]

for i in range(len(plist)):
  print("\ntemplate:", plist[i])
  template = ImgDatabase + "/" + plist[i] + "/" + "1.pgm"
  # 顔画像ファイルの読み込み
  img1 = cv2.imread(template, cv2.IMREAD_GRAYSCALE)
  # 画素値をfloat型に変換
  img1 = img1.astype(np.float)

  # 本人の顔画像どうしを比較
  print("genuine:")
  # print(plist[i])
  res1 = np.array([])
  for j in range(2,Ndat+1):
    genuine = ImgDatabase + "/" + plist[i] + "/" + str(j) + ".pgm"
    img2 = cv2.imread(genuine, cv2.IMREAD_GRAYSCALE)
    img2 = img2.astype(np.float)
    # 比較結果（画像間の距離）を順次res1に追加
    res1 = np.append(res1, distance(img1, img2))
  # 平均距離を表示
  print("mean distance (%d) = %f" % (res1.size, np.average(res1)))
  # res1の内容をgdistに追加
  gdist = np.append(gdist, res1)

  # 本人と他人の顔画像を比較
  print("others:")
  res2 = np.array([])
  for k in range(len(plist)):
    if(k != i):
      # print(plist[k])
      for l in range(1,Ndat+1):
        other = ImgDatabase + "/" + plist[k] + "/" + str(l) + ".pgm"
        img2 = cv2.imread(other, cv2.IMREAD_GRAYSCALE)
        # 比較結果（画像間の距離）を順次res2に追加
        res2 = np.append(res2, distance(img1, img2))
  # 平均距離を表示
  print("mean distance (%d) = %f" % (res2.size, np.average(res2)))
  # res2の内容をodistに追加
  odist = np.append(odist, res2)

# gdistとodistの内容をファイルに出力
np.savetxt("gdist.dat", gdist)
np.savetxt("odist.dat", odist)
