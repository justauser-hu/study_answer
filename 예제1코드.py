import random
import math as m
import time

startTime = time.time()
x_a=[]
y_a=[]
x_b=[]
y_b=[]
x_c=[]
y_c=[]
maxDistance=0
for i in range(100):
    x_a.append(random.randrange(100))
    y_a.append(random.randrange(100))
    x_b.append(random.randrange(100))
    y_b.append(random.randrange(100))
    x_c.append(random.randrange(100))
    y_c.append(random.randrange(100))
for i in range(100):
    vecx_ba = float(x_a[i]-x_b[i])
    vecy_ba = float(y_a[i]-y_b[i])
    vecx_bc = float(x_c[i]-x_b[i])
    vecy_bc = float(y_c[i]-y_b[i])
    len_ba = m.sqrt(vecx_ba**2+vecy_ba**2)
    len_bc = m.sqrt(vecx_bc**2+vecy_bc**2)
    angle_b = m.acos((vecx_ba*vecx_bc+vecy_ba*vecy_bc)/len_ba/len_bc) ## angle in radian
    len_BtoAC = m.sin(angle_b)*len_ba*len_bc/m.sqrt((x_a[i]-x_c[i])**2+(y_a[i]-y_c[i])**2)
    if len_BtoAC>maxDistance :
        maxDistance=len_BtoAC
    print("(%d,%d), (%d,%d), (%d,%d), Angle B is %f, Length from B to AC is %f"%(x_a[i], y_a[i], x_b[i], y_b[i], x_c[i], y_c[i], angle_b, len_BtoAC))
endTime=time.time()
print('{}ms'.format((endTime-startTime)*1000))
print("Maximum Distance is %f"%(maxDistance))