import numpy as np
from numbapro import cuda, jit, float32, int32
from numba import *
import time 

bpg = 1024
tpb = 256

n = bpg * tpb * 32

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
	


def mergeChunks(s1, e1, s2, e2, B):
	mergeSize = (e1-s1) + (e2-s2)
	merged = np.empty(mergeSize)
	k = 0
	i = s1
	j = s2
	while(i < e1 and j < e2):
		if B[i] <= B[j]:
			merged[k] = B[i]
			i += 1
			k += 1
		else:
			merged[k] = B[j]
			j += 1
			k += 1
	while(i < e1):
		merged[k] = B[i]
		i += 1
		k += 1
	while(j < e2):
		merged[k] = B[j]
		k += 1
		j += 1

	k = 0
	for index in range(s1, e2, 1): 
		B[index] = merged[k]
		k += 1



disorder = np.array(np.random.random(n), dtype=np.float32)
timer = time.time()

dA = cuda.to_device(disorder)
bbSort[(bpg, n/(bpg*tpb)), tpb](dA)
dA.to_host()


runtime = time.time() - timer
print("Bubble sort takes %fs" % runtime)

sortedSize = tpb;

while(sortedSize != n):
	chunks = n / sortedSize
	for i in range(0, chunks, 2):
		mergeChunks(i*sortedSize, (i+1)*sortedSize, (i+1)*sortedSize, (i+2)*sortedSize, disorder)
	sortedSize = sortedSize * 2

runtime = time.time() - timer
print("Total takes %fs" % runtime)

f = open("sorted.txt", "w")
for i in disorder:
	print >> f, "%f" % i
f.close()