# Error rate calculation

import os
import numpy as np
import matplotlib.pyplot as plt

# distance file for genuine
gdistfile = "gdist.dat"
# distance file for others
odistfile = "odist.dat"
# min. threshold
minth = 500
# max. threshold
maxth = 5000

gdist = np.loadtxt(gdistfile)
odist = np.loadtxt(odistfile)

#print(gdist)
#print(odist)
result = np.empty((0,3), float)

for th in range(minth, maxth, 100):
  frr = np.sum(gdist > th) / gdist.size
  far = np.sum(odist <= th) / odist.size
  result = np.append(result, np.array([[th,frr,far]]), axis=0)
# print(result)

x = result[:,0:1]
y1 = result[:,1:2]
y2 = result[:,2:3]

plt.style.use('default')
  
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.plot(x, y1, label='FRR', marker='o')
ax.plot(x, y2, label='FAR', marker='o')

ax.legend()
ax.set_xlabel("threshold th")
ax.set_ylabel("error rate")

plt.show()
