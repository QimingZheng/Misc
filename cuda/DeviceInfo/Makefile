main: main.cu
	nvcc -I /usr/local/cuda/samples/common/inc -o main main.cu

test: test.cu
	nvcc -I /usr/local/cuda/samples/common/inc -o test test.cu

clean:
	rm -f main test
