CFILES:=$(shell find -L . -type f -name '*.c' | grep -v "./venv/")
.PHONY: all clean

all: clean
	gcc $(CFILES) -o cnn -Iinclude/ -lm -O3 -Wno-incompatible-function-pointer-types

fast: clean
	gcc $(CFILES) -o cnn -Iinclude/ -lm -Ofast

clean:
	rm -rf ./cnn

asm:
	gcc -S $(CFILES) -Iinclude/

check:
	gcc $(CFILES) -o cnn -Iinclude/ -lm -Og -ggdb3 -DVALGRIND
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out.txt ./cnn
