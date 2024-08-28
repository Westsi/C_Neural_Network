CFILES:=$(shell find -L . -type f -name '*.c' | grep -v "./venv/")
.PHONY: all clean

all: clean
	gcc $(CFILES) -o cnn -Iinclude/ -lm -O3 -Wno-incompatible-function-pointer-types -lpthread

fast: clean
	gcc $(CFILES) -o cnn -Iinclude/ -lm -Ofast -Wno-incompatible-function-pointer-types -lpthread

clean:
	rm -rf ./cnn

asm:
	gcc -S $(CFILES) -Iinclude/ -lm -Wno-incompatible-function-pointer-types -lpthread

check:
	gcc $(CFILES) -o cnn -Iinclude/ -lm -Og -ggdb3 -DVALGRIND -Wno-incompatible-function-pointer-types -lpthread
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out.txt ./cnn

prof: clean
	gcc $(CFILES) -o cnn -Iinclude/ -lm -Ofast -Wno-incompatible-function-pointer-types -pg -g -lpthread
	./cnn
	gprof ./cnn ./gmon.out -l > gmon.txt