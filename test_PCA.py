import numpy as np
N = 20
n = 5
a = np.random.rand(N,n)
b = np.zeros((N,n+1))
c = np.random.randint(2, size=N)
print(c)
print(a.shape)

print(a)
print(b)

b[:,:-1] = a
b[:,-1] = c

print(b)

A = np.zeros(30, dtype=np.float32)
B = np.zeros(30, dtype=np.int32)
C = np.zeros(30, dtype=np.float32)

res = np.rec.fromarrays([A,B,C], names='a,b,c')

print(res)