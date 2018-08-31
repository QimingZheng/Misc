NVCC:=nvcc 
NVCCFLAGS:= -g -G -std=c++11 -I /usr/local/cuda/samples/common/inc -lcudart

main: main.cu
	$(NVCC) -o $@ $<  $(NVCCFLAGS)

naive: naive.cu
	$(NVCC) -o $@ $<  $(NVCCFLAGS)


clean:
	rm -f main naive
