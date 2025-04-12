# Makefile for building MultiDevOneP
# Define compiler
CXX := g++
# Compiler flags
CXXFLAGS := -g -I../build/include/ -I/usr/local/cuda/include
# Source file
SRC_ONEP := MultiDevOneP.cc
# Output binary
TARGET_ONEP := MultiDevOneP
# Linker flags for MultiDevOneP
LDFLAGS_ONEP := -O0 -I ../build/include -L../build/lib/ -L/usr/local/cuda/lib64 -lnccl
# Build target
all: $(TARGET_ONEP)
$(TARGET_ONEP): $(SRC_ONEP)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS_ONEP) -o $@
# Clean target
clean:
	rm -f $(TARGET_ONEP)
# PHONY targets
.PHONY: all clean