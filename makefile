CFILES:=$(shell find -L . -type f -name '*.c')
.PHONY: all clean

all: clean
	clang $(CFILES) -o cnn -Iinclude/ -lm

clean:
	rm -rf ./cnn

asm:
	clang -S $(CFILES) -Iinclude/