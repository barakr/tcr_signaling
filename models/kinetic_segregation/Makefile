CC = clang
CXX = clang++
CFLAGS = -O2 -Wall -Wextra -std=c11
CXXFLAGS = -O2 -Wall -Wextra -std=c++20
OBJCFLAGS = -O2 -Wall -Wextra -fobjc-arc
FRAMEWORKS = -framework Metal -framework Foundation
LDFLAGS = $(FRAMEWORKS)

SRCDIR = src
BUILDDIR = build
TARGET = ks_gpu

SOURCES_C = $(SRCDIR)/rng.c $(SRCDIR)/potentials.c $(SRCDIR)/simulation.c
SOURCES_M = $(SRCDIR)/metal_engine.m
SOURCES_CXX = $(SRCDIR)/main.cpp
OBJECTS_C = $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(SOURCES_C))
OBJECTS_M = $(patsubst $(SRCDIR)/%.m,$(BUILDDIR)/%.o,$(SOURCES_M))
OBJECTS_CXX = $(patsubst $(SRCDIR)/%.cpp,$(BUILDDIR)/%.o,$(SOURCES_CXX))
OBJECTS = $(OBJECTS_C) $(OBJECTS_M) $(OBJECTS_CXX)

# Shared library for Python ctypes testing of C functions
TESTLIB = $(BUILDDIR)/libks_potentials.dylib

# Pre-compiled Metal shader
METALLIB = $(SRCDIR)/shaders.metallib
METALAIR = $(BUILDDIR)/shaders.air

# Check if Metal toolchain is available
METAL_AVAILABLE := $(shell xcrun -sdk macosx --find metal 2>/dev/null && echo yes || echo no)

.PHONY: all clean testlib metallib

all: $(TARGET)
ifeq ($(METAL_AVAILABLE),yes)
all: $(METALLIB)
endif

metallib: $(METALLIB)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^

$(BUILDDIR)/%.o: $(SRCDIR)/%.c $(SRCDIR)/*.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILDDIR)/%.o: $(SRCDIR)/%.m $(SRCDIR)/*.h | $(BUILDDIR)
	$(CC) $(OBJCFLAGS) -c -o $@ $<

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp $(SRCDIR)/*.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -c -o $@ $<

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Pre-compile Metal shader to .metallib for faster init
# Note: -I$(SRCDIR) needed so shaders.metal can #include "ks_physics.h"
$(METALLIB): $(SRCDIR)/shaders.metal $(SRCDIR)/ks_physics.h | $(BUILDDIR)
	xcrun -sdk macosx metal -c -I$(SRCDIR) -o $(METALAIR) $<
	xcrun -sdk macosx metallib -o $@ $(METALAIR)

# Build a shared library exporting potentials + rng for Python test access via ctypes
testlib: $(TESTLIB)

$(TESTLIB): $(SRCDIR)/potentials.c $(SRCDIR)/rng.c $(SRCDIR)/potentials.h $(SRCDIR)/rng.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -shared -o $@ $(SRCDIR)/potentials.c $(SRCDIR)/rng.c -lm

clean:
	rm -rf $(BUILDDIR) $(TARGET) $(METALLIB)
