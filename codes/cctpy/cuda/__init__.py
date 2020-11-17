import sys

try:
    import pycuda
except ImportError:
    print("使用 CUDA 必须正确安装 pycuda")
    sys.exit(-1)
