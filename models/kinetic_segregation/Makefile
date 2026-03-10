# Thin wrapper around CMake — preserves make / make testlib / make pdf / make clean.
# For direct CMake usage:  cmake -S . -B build && cmake --build build

.PHONY: all testlib pdf metallib clean

all:
	@cmake -S . -B build --log-level=WARNING 2>&1 | grep -v "^$$" || true
	@cmake --build build

testlib:
	@cmake -S . -B build --log-level=WARNING 2>&1 | grep -v "^$$" || true
	@cmake --build build --target ks_potentials

pdf:
	@cmake -S . -B build --log-level=WARNING 2>&1 | grep -v "^$$" || true
	@cmake --build build --target pdf

metallib:
	@cmake -S . -B build --log-level=WARNING 2>&1 | grep -v "^$$" || true
	@cmake --build build --target metallib

clean:
	rm -rf build ks_gpu src/shaders.metallib
