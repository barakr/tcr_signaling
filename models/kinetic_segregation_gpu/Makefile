CC = clang
CFLAGS = -O2 -Wall -Wextra -std=c11
OBJCFLAGS = -O2 -Wall -Wextra -fobjc-arc
FRAMEWORKS = -framework Metal -framework Foundation
LDFLAGS = $(FRAMEWORKS)

SRCDIR = src
BUILDDIR = build
TARGET = ks_gpu

SOURCES_C = $(SRCDIR)/rng.c $(SRCDIR)/potentials.c $(SRCDIR)/simulation.c
SOURCES_M = $(SRCDIR)/main.m $(SRCDIR)/metal_engine.m
OBJECTS_C = $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(SOURCES_C))
OBJECTS_M = $(patsubst $(SRCDIR)/%.m,$(BUILDDIR)/%.o,$(SOURCES_M))
OBJECTS = $(OBJECTS_C) $(OBJECTS_M)

# Shared library for Python ctypes testing of C functions
TESTLIB = $(BUILDDIR)/libks_potentials.dylib

.PHONY: all clean testlib

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^

$(BUILDDIR)/%.o: $(SRCDIR)/%.c $(SRCDIR)/*.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILDDIR)/%.o: $(SRCDIR)/%.m $(SRCDIR)/*.h | $(BUILDDIR)
	$(CC) $(OBJCFLAGS) -c -o $@ $<

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Build a shared library exporting potentials + rng for Python test access via ctypes
testlib: $(TESTLIB)

$(TESTLIB): $(SRCDIR)/potentials.c $(SRCDIR)/rng.c $(SRCDIR)/potentials.h $(SRCDIR)/rng.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -shared -o $@ $(SRCDIR)/potentials.c $(SRCDIR)/rng.c -lm

clean:
	rm -rf $(BUILDDIR) $(TARGET)
