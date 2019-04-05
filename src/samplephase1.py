import numpy as np
import ctypes
from ctypes import *

class MATRIX(Structure):
    _fields_ = [("m",POINTER(c_float)),("x", c_int),("y", c_int)]

# extract cuda_sum function pointer in the shared object cuda_sum.so
def get_cuda_sum():
    dll = ctypes.CDLL('./phase2.so', mode=ctypes.RTLD_GLOBAL)
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

#
#
#
#
# extract cuda_sum function pointer in the shared object cuda_sum.so
def get_cuda_transpose():
    dll = ctypes.CDLL('./phase2.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.transpose
    func.argtypes = [MATRIX]
    func.restype = MATRIX
    return func

# create __cuda_sum function with get_cuda_sum()
__cuda_transpose = get_cuda_transpose()

# convenient python wrapper for __cuda_sum
# it does all job with types convertation
# from python ones to C++ ones
def cuda_transpose(a):

    a_c = MATRIX(a.ctypes.data_as(POINTER(c_float)) ,c_int(a.shape[0]),c_int(a.shape[1]))
    c_p = __cuda_transpose(a_c)
    shape = c_p.x * c_p.y
    arr = np.ctypeslib.as_array(obj=c_p.m,shape = (shape,))
    arr = np.reshape(arr,(c_p.x,c_p.y))
    return arr
#
#
#
#

#
#
#
#
# extract cuda_sum function pointer in the shared object cuda_sum.so
def get_cuda_matmul():
    dll = ctypes.CDLL('./phase2.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.matmul
    func.argtypes = [MATRIX,MATRIX]
    func.restype = MATRIX
    return func

# create __cuda_sum function with get_cuda_sum()
__cuda_matmul = get_cuda_matmul()

# convenient python wrapper for __cuda_sum
# it does all job with types convertation
# from python ones to C++ ones
def cuda_matmul(a,b):

    a_c = MATRIX(a.ctypes.data_as(POINTER(c_float)) ,c_int(a.shape[0]),c_int(a.shape[1]))
    b_c = MATRIX(b.ctypes.data_as(POINTER(c_float)) ,c_int(b.shape[0]),c_int(b.shape[1]))
    c_p = __cuda_matmul(a_c,b_c)
    shape = c_p.x * c_p.y
    arr = np.ctypeslib.as_array(obj=c_p.m,shape = (shape,))
    arr = np.reshape(arr,(c_p.x,c_p.y))
    return arr
#
#
#
#

#
#
#
#
# extract cuda_sum function pointer in the shared object cuda_sum.so
def get_cuda_add_mat():
    dll = ctypes.CDLL('./phase2.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.add_mat
    func.argtypes = [MATRIX,MATRIX]
    func.restype = MATRIX
    return func

# create __cuda_sum function with get_cuda_sum()
__cuda_add_mat = get_cuda_add_mat()

# convenient python wrapper for __cuda_sum
# it does all job with types convertation
# from python ones to C++ ones
def cuda_add_mat(a,b):

    a_c = MATRIX(a.ctypes.data_as(POINTER(c_float)) ,c_int(a.shape[0]),c_int(a.shape[1]))
    b_c = MATRIX(b.ctypes.data_as(POINTER(c_float)) ,c_int(b.shape[0]),c_int(b.shape[1]))
    c_p = __cuda_add_mat(a_c,b_c)
    shape = c_p.x * c_p.y
    arr = np.ctypeslib.as_array(obj=c_p.m,shape = (shape,))
    arr = np.reshape(arr,(c_p.x,c_p.y))
    return arr
#
#
#
#

#
#
#
#
# extract cuda_sum function pointer in the shared object cuda_sum.so
def get_cuda_multiply():
    dll = ctypes.CDLL('./phase2.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.multiply
    func.argtypes = [MATRIX,MATRIX]
    func.restype = MATRIX
    return func

# create __cuda_sum function with get_cuda_sum()
__cuda_multiply = get_cuda_multiply()

# convenient python wrapper for __cuda_sum
# it does all job with types convertation
# from python ones to C++ ones
def cuda_multiply(a,b):

    a_c = MATRIX(a.ctypes.data_as(POINTER(c_float)) ,c_int(a.shape[0]),c_int(a.shape[1]))
    b_c = MATRIX(b.ctypes.data_as(POINTER(c_float)) ,c_int(b.shape[0]),c_int(b.shape[1]))
    c_p = __cuda_multiply(a_c,b_c)
    shape = c_p.x * c_p.y
    arr = np.ctypeslib.as_array(obj=c_p.m,shape = (shape,))
    arr = np.reshape(arr,(c_p.x,c_p.y))
    return arr
#
#
#
#

#
#
#
#
# extract cuda_sum function pointer in the shared object cuda_sum.so
def get_cuda_sub_mat():
    dll = ctypes.CDLL('./phase2.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.sub_mat
    func.argtypes = [MATRIX,MATRIX]
    func.restype = MATRIX
    return func

# create __cuda_sum function with get_cuda_sum()
__cuda_sub_mat = get_cuda_sub_mat()

# convenient python wrapper for __cuda_sum
# it does all job with types convertation
# from python ones to C++ ones
def cuda_sub_mat(a,b):

    a_c = MATRIX(a.ctypes.data_as(POINTER(c_float)) ,c_int(a.shape[0]),c_int(a.shape[1]))
    b_c = MATRIX(b.ctypes.data_as(POINTER(c_float)) ,c_int(b.shape[0]),c_int(b.shape[1]))
    c_p = __cuda_sub_mat(a_c,b_c)
    shape = c_p.x * c_p.y
    arr = np.ctypeslib.as_array(obj=c_p.m,shape = (shape,))
    arr = np.reshape(arr,(c_p.x,c_p.y))
    return arr
#
#
#
#

#
#
#
#
# extract cuda_sum function pointer in the shared object cuda_sum.so
def get_cuda_divide():
    dll = ctypes.CDLL('./phase2.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.divide
    func.argtypes = [MATRIX,MATRIX]
    func.restype = MATRIX
    return func

# create __cuda_sum function with get_cuda_sum()
__cuda_divide = get_cuda_divide()

# convenient python wrapper for __cuda_sum
# it does all job with types convertation
# from python ones to C++ ones
def cuda_divide(a,b):

    a_c = MATRIX(a.ctypes.data_as(POINTER(c_float)) ,c_int(a.shape[0]),c_int(a.shape[1]))
    b_c = MATRIX(b.ctypes.data_as(POINTER(c_float)) ,c_int(b.shape[0]),c_int(b.shape[1]))
    c_p = __cuda_divide(a_c,b_c)
    shape = c_p.x * c_p.y
    arr = np.ctypeslib.as_array(obj=c_p.m,shape = (shape,))
    arr = np.reshape(arr,(c_p.x,c_p.y))
    return arr
#
#
#
#

# testing, sum of two arrays of ones and output head part of resulting array
if __name__ == '__main__':

    a = np.random.rand(4,3).astype('float32') * 100
    b = np.random.rand(4,3).astype('float32') * 100


    print(a)
    print(b)

    c_p = cuda_add_mat(a,b)

    print(c_p.shape)
    print(c_p)



    #print c[:10]
