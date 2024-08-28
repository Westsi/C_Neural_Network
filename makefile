CFILES:=$(shell find -L . -type f -name '*.c' | grep -v "./venv/")
.PHONY: all clean

all: clean
	nvcc -c nvcalls.o nvcalls.cu
	gcc $(CFILES) nvcalls.o -o cnn -Iinclude/ -lm -O3 -L/usr/local/cuda/lib64 -lcudart -Wno-incompatible-function-pointer-types -lpthread

fast: clean
	nvcc -c nvcalls.o nvcalls.cu
	gcc $(CFILES) nvcalls.o -o cnn -Iinclude/ -lm -Ofast -L/usr/local/cuda/lib64 -lcudart -Wno-incompatible-function-pointer-types -lpthread

clean:
	rm -rf ./cnn
	rm -rf ./nvcalls.o

check:
	nvcc -c nvcalls.o nvcalls.cu
	gcc $(CFILES) -o cnn -Iinclude/ -lm -Og -ggdb3 -DVALGRIND -L/usr/local/cuda/lib64 -lcudart -Wno-incompatible-function-pointer-types -lpthread
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out.txt ./cnn

prof: clean
	gcc $(CFILES) -o cnn -Iinclude/ -lm -Ofast -Wno-incompatible-function-pointer-types -pg -g -L/usr/local/cuda/lib64 -lcudart -Wno-incompatible-function-pointer-types -lpthread
	./cnn
	gprof ./cnn ./gmon.out -l > gmon.txt