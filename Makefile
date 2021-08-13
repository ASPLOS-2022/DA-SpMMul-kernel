GPUARCH = sm_70
NVCC = nvcc
CCFLAG = -O3 -std=c++11 -arch=$(GPUARCH) -lcusparse

all: spmvspmm trace.spmvspmm.out spmvspmm.mps.out

spmvspmm: spmvspmm_benchmark.cu *.h
	$(NVCC) $(CCFLAG) -o $@ $<

clean: 
	rm -f spmvspmm trace.spmvspmm.out