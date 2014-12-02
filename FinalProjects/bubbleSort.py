import numpy as np 
import time

bpg = 1024
tpb = 256
chunkSize = 32
n = bpg * tpb * chunkSize


def bbSort(A, BlkId):
	leng = len(A)

	si = chunkSize * BlkId
	ei = chunkSize * (BlkId + 1)

	for i in range(ei, si, -1):
		for j in range(si, i - 1, 1):
			if A[j] > A[j+1] :
				temp = A[j]
				A[j] = A[j+1]
				A[j+1] = temp
	
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

origin = np.array(np.random.random(n), dtype=np.float32)
disorder = origin.copy()

timer = time.time()

for i in range (n/chunkSize):
	bbSort(disorder, i)

runtime = time.time() - timer
print("Bubble sort takes %fs" % runtime)

sortedSize = chunkSize;

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