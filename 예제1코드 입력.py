import numpy as np
import time
from matplotlib import pyplot as plt

n=3

def vec_mag(v) :
    return np.sqrt(np.sum(v**2, axis=1))

def angleB(A,B,C) :
    return np.arccos(np.sum((A-B)*(C-B), axis=1)/vec_mag(A-B)/vec_mag(C-B))

def disAtoBC(A,B,C) :
    areaX2 = np.sin(angleB(A,B,C))*vec_mag(A-B)*vec_mag(C-B)
    return areaX2/vec_mag(C-B)

a= np.zeros((n,2))
b= np.zeros((n,2))
c= np.zeros((n,2))

print("input as P x,y")

for i in range(n) :
    s= input("A{} ".format(i+1))
    a[i,:] = [float(s.split(',')[0]), float(s.split(',')[1])]
    s = input("B{} ".format(i+1))
    b[i, :] = [float(s.split(',')[0]), float(s.split(',')[1])]
    s = input("C{} ".format(i+1))
    c[i, :] = [float(s.split(',')[0]), float(s.split(',')[1])]

startTime = time.time()

arrPoint = np.concatenate((a,b,c), axis=1)

print("\n")
print("Points :\n", arrPoint)
print("\n")
print("angleB :\n", np.reshape(angleB(a,b,c), (n,1)))
print("\n")
print("distance from A to BC :\n", np.reshape(disAtoBC(a,b,c), (n,1)))
print("\n")

endTime=time.time()

print('Running Time : %.3f ms'%((endTime-startTime)*1000))

plt.xlim(0,100)
plt.ylim(0,100)
for i in range(n) :
    plt.arrow(b[i,0], b[i,1], a[i,0]-b[i,0], a[i,1]-b[i,1], head_width=2, length_includes_head=True)
    plt.arrow(b[i, 0], b[i, 1], c[i, 0] - b[i, 0], c[i, 1] - b[i, 1], head_width=2, length_includes_head=True)
plt.show()