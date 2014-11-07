# Fast Python on CUDA-capable GPUs

Python is one of the most widely used dynamic languages these days and its flexibility and expressive syntax make it a great language for quick prototyping. But as a interpreted language, Python is considered as poor performance for massive data processes and computation. There are a few ways to accelarate the computation speed of Python and using CUDA-capable GPUs is a brilliant idea, which combines the Python code with powerful computating ability of GPU. And our final project focuses on the NumbaPro, a Python compiler that can compile Python code for execution on CUDA-capable GPUS and multicore CPUs.

## Intro to NumbaPro

NumbaPro compiler targets multi-core CPU and GPUs directly from simple Python syntax, which enables easily move vectorized NumPy functions to the GPU and has multiple CUDA device support.

Except this, NumbaPro provides a Python interface to CUDA cuBLAS (dense linear algebra), cuFFT (Fast Fourier Transform), and cuRAND (random number generation)libraries. And its CUDA Python API provides explicit control over data transfer and CUDA streams.

## Final Project

In the first place, we are going to get familiar with NumbaPro and then use some practical examples to test its performance speed-up and results compared with single thread, CPU execution.

The remaining part of this project focuses on coming up with more advanced decorators, which optimizes the communication between Python code and CUDA-capable GPUs.

More details about NumbaPro:
---http://devblogs.nvidia.com/parallelforall/numbapro-high-performance-python-cuda-acceleration/
