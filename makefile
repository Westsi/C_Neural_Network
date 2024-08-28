CFILES:=$(shell find -L . -type f -name '*.c' | grep -v "./venv/")
.PHONY: all clean

all: clean
	gcc $(CFILES) -o cnn -Iinclude/ -lm -O3 -Wno-incompatible-function-pointer-types

fast: clean
	gcc $(CFILES) -o cnn -Iinclude/ -lm -Ofast -Wno-incompatible-function-pointer-types

clean:
	rm -rf ./cnn

asm:
	gcc -S $(CFILES) -Iinclude/ -Wno-incompatible-function-pointer-types

check:
	gcc $(CFILES) -o cnn -Iinclude/ -lm -Og -ggdb3 -DVALGRIND -Wno-incompatible-function-pointer-types
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out.txt ./cnn
