import numpy as np
import ctypes
from ctypes import *

class MATRIX(Structure):
    _fields_ = [("m",POINTER(c_float)),("x", c_int),("y", c_int)]

# extract cuda_sum function pointer in the shared object cuda_sum.so
def get_cuda_sum():
    dll = ctypes.CDLL('./phase1.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.add_mat
    func.argtypes = [MATRIX,MATRIX]
    func.restype = MATRIX
    return func

# create __cuda_sum function with get_cuda_sum()
__cuda_sum = get_cuda_sum()

# convenient python wrapper for __cuda_sum
# it does all job with types convertation
# from python ones to C++ ones
def cuda_sum(a, b):

    c_p = __cuda_sum(a, b)
    shape = c_p.x * c_p.y
    arr = np.ctypeslib.as_array(obj=c_p.m,shape = (shape,))
    arr = np.reshape(arr,(c_p.x,c_p.y))
    return arr

# testing, sum of two arrays of ones and output head part of resulting array
if __name__ == '__main__':

    x = 10 #try 20
    y = 100

    size = int(x*y)

    a = np.ones(size).astype('float32')
    b = np.ones(size).astype('float32')

    a_c = MATRIX(a.ctypes.data_as(POINTER(c_float)) ,c_int(x),c_int(y))
    b_c = MATRIX(b.ctypes.data_as(POINTER(c_float)) ,c_int(x),c_int(y))

    #print(a_c.x)

    c_p = cuda_sum(a_c, b_c)

    print(c_p)

    #print c[:10]
