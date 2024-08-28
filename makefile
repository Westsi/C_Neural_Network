CFILES := $(shell find -L . -type f -name '*.c' | grep -v "./venv/")
OBJFILES := $(patsubst %.c,%.o,$(CFILES))

TARGET := cnn

CC := gcc
NVCC := nvcc

CUDAFILES := $(shell find -L . -type f -name '*.cu' | grep -v "./venv/")
CUOBJFILES := $(patsubst %.cu,%.o,$(CUDAFILES))

INCLUDE := -Iinclude/
LDFLAGS := -L/usr/local/cuda/lib64

LIBS := -lcudart -lm -lpthread

OPTIM := -O3

# Flags for Valgrind Checking
VGFLAGS := -ggdb3 -DVALGRIND

PROFFLAGS := $(OPTIM) -pg -g

CFLAGS := $(OPTIM) -ffast-math -Wno-incompatible-function-pointer-types -Wno-unused-result -Wno-incompatible-pointer-types $(INCLUDE)
NVCCFLAGS := $(OPTIM) $(INCLUDE) --use_fast_math

.PHONY: all clean check prof

all: clean $(TARGET)

clean:
	rm -rf $(TARGET) $(CUOBJFILES) $(OBJFILES)
	rm -rf ./gmon.out

$(TARGET): $(CUOBJFILES) $(OBJFILES)
	$(CC) $(OBJFILES) $(CUOBJFILES) -o $(TARGET) $(CFLAGS) $(LDFLAGS) $(LIBS)

%.o: %.c
	$(CC) -c $< -o $@ $(CFLAGS)

%.o: %.cu
	$(NVCC) -c $< -o $@ $(NVCCFLAGS)


check: OPTIM = -Og
check: CFLAGS += $(VGFLAGS)
check: clean $(TARGET)
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out.txt ./$(TARGET)

prof: CFLAGS += $(PROFFLAGS)
prof: NVCCFLAGS += $(PROFFLAGS)
prof: NVCCFLAGS += -lineinfo
prof: clean $(TARGET)
	./$(TARGET)
	gprof ./$(TARGET) ./gmon.out -l > gmon.txt

# check:
# 	nvcc -c nvcalls.o nvcalls.cu
# 	gcc $(CFILES) -o cnn -Iinclude/ -lm -Og -ggdb3 -DVALGRIND -L/usr/local/cuda/lib64 -lcudart -Wno-incompatible-function-pointer-types -lpthread
# 	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out.txt ./cnn

# prof: clean
# 	gcc $(CFILES) -o cnn -Iinclude/ -lm -Ofast -Wno-incompatible-function-pointer-types -pg -g -L/usr/local/cuda/lib64 -lcudart -Wno-incompatible-function-pointer-types -lpthread
# 	./cnn
# 	gprof ./cnn ./gmon.out -l > gmon.txt