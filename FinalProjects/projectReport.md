#Write cuda directly in Python code

Our previous study focused on the 'vectorize' automatically optimization and CUDA libraries host API, and both of them show dramatical speed-up for calculation-intensive python code. In this part of project, we will demenstrate how to write cuda kernel function in pure python code and test its performance. 

Specifically, we have tried two simple examples: bubble sort and blowfish data encryption.

##Bubble sort

The reason why we choose bubble sort is because that it could take very large array for calculation and the sort part can be easily paralleled. 

The cpu serial version is straightforward and time-consuming: divide input array into chunks and after sorting, merge them again. And the gpu part realizes the cuda parallelization: 

```python

dA = cuda.to_device(disorder)
bbSort[bpg, tpb](dA)
dA.to_host()

...

@cuda.jit(argtypes=[float32[:]], target='gpu')
def bbSort(A):
    threadID = cuda.grid(1)
    si = chunkSize * threadID
    ei = chunkSize * (threadID + 1)
    for i in range(ei, si, -1):
        for j in range(si, i - 1, 1):
            if A[j] > A[j+1] :
                temp = A[j]
                A[j] = A[j+1]
                A[j+1] = temp

```

Once the array is transferred from cpu to gpu, the kernel part is able to be launched with specification of number of blocks per grid and number of threads per block. After the calculation, the sorted array will be transferred back to cpu then merged.

One remarkable feature of CUDA code is to use shared memory for acceleration. Here is the kernel with shared memory:

```python

@cuda.jit(argtypes=[float32[:]], target='gpu')
def bbSort(A):
    blockID = cuda.blockIdx.y * cuda.gridDim.x + cuda.blockIdx.x
    totalID = blockID * cuda.blockDim.x + cuda.threadIdx.x
    threadID = cuda.threadIdx.x
    sA = cuda.shared.array(shape=(256), dtype=float32)
    swapped = cuda.shared.array(shape=(1), dtype=int32)
    swapped[0] = 1
    sA[threadID] = A[totalID]
    cuda.syncthreads()
    while  swapped[0]:
        cuda.syncthreads()
        swapped[0] = 0
        if((threadID%2==0) and (sA[threadID] > sA[threadID+1])):
            swapped[0] = 1
            temp = sA[threadID]
            sA[threadID] = sA[threadID+1]
            sA[threadID+1] = temp
        cuda.syncthreads()
        if((threadID%2==1) and (sA[threadID] > sA[threadID+1]) and (threadID != cuda.blockDim.x - 1)):
            swapped[0] = 1
            temp = sA[threadID]
            sA[threadID] = sA[threadID+1]
            sA[threadID+1] = temp
        cuda.syncthreads()
    cuda.syncthreads()
    A[totalID] = sA[threadID]
    cuda.syncthreads()

```

As can be seen from the example, numbapro enables python codes with most of its original functionality while keeps the simplicity as the same time.

##Blowfish encryption

Blowfish enciphers a picture read in bytes with multiple calculation, and the decipher part can be used to verify the correctness of encryption.




