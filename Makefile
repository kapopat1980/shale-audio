# Build the two shared libraries used by the Python package, and the portable C benchmark.
CC ?= gcc
CFLAGS ?= -O3 -ffp-contract=off -fPIC -Wall

all: lib/libshale_ascon.so lib/libshale_codec.so shale_bench

lib/libshale_ascon.so: csrc/ascon.c
	mkdir -p lib && $(CC) $(CFLAGS) -shared $< -o $@

lib/libshale_codec.so: csrc/codec.c
	mkdir -p lib && $(CC) $(CFLAGS) -shared $< -o $@ -lm

shale_bench: csrc/bench_main.c csrc/codec.c csrc/ascon.c
	$(CC) -O3 -ffp-contract=off $^ -o $@ -lm

test: all
	python3 tests/test_ascon.py
	python3 tests/test_selfcontained.py

clean:
	rm -f lib/*.so shale_bench

.PHONY: all test clean
