# Makefile for OmniDB CoProcessor on Linux (Kernel Scheduler)
# Target: Linux with Intel CPU/GPU using PoCL (Portable Computing Language)

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall -Wno-write-strings -Wno-unused-variable -Wno-comment -fpermissive -fPIC -D__linux__ -DCL_TARGET_OPENCL_VERSION=300 -fopenmp
CFLAGS = -O2 -Wall -fPIC -D__linux__

# OpenCL configuration
OPENCL_CFLAGS = $(shell pkg-config --cflags OpenCL 2>/dev/null || echo "-I/usr/include")
OPENCL_LIBS = $(shell pkg-config --libs OpenCL 2>/dev/null || echo "-lOpenCL")

# KernelEngine shared library path
KERNEL_ENGINE_DIR = ../KernelEngine_KernelSchedule/DLL_OpenCL

# Include paths
INCLUDES = -I./CoProcessor -I./MyLib -I./TonyLib -I./MianLib $(OPENCL_CFLAGS)

# Libraries
# Link against KernelEngine static library (single executable, no .so needed)
# Use --whole-archive to pull in all symbols from static lib
LIBS = -L$(KERNEL_ENGINE_DIR) -Wl,--whole-archive -lomnidb_opencl -Wl,--no-whole-archive $(OPENCL_LIBS) -lpthread -lm -lgomp

# Build directory
BUILD_DIR = build

# CoProcessor source files (in CoProcessor/ directory)
COP_SOURCES = \
	CoProcessor/CoProcessorApp.cpp \
	CoProcessor/DynamicQueryProcessor.cpp \
	CoProcessor/HandShaking.cpp \
	CoProcessor/Database.cpp \
	CoProcessor/db.cpp \
	CoProcessor/QueryPlanTree.cpp \
	CoProcessor/QueryPlanNode.cpp \
	CoProcessor/PredicateTree.cpp \
	CoProcessor/ThreadOp.cpp \
	CoProcessor/BinaryThreadOp.cpp \
	CoProcessor/SingularThreadOp.cpp \
	CoProcessor/SortThreadOp.cpp \
	CoProcessor/GroupByThreadOp.cpp \
	CoProcessor/Co_Hj.cpp \
	CoProcessor/Co_Inlj.cpp \
	CoProcessor/CO_Ninlj.cpp \
	CoProcessor/Co_Sort.cpp \
	CoProcessor/Co_Smj.cpp \
	CoProcessor/UInterface.cpp \
	CoProcessor/QP_Utility.cpp \
	CoProcessor/SDKCommmon.cpp \
	CoProcessor/CoProcessorTest.cpp \
	CoProcessor/CPU_Stubs.cpp

# Object files (in build directory)
OBJECTS = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(COP_SOURCES))

# Executable name
TARGET = coprocessor_bin

# Default target
all: $(TARGET)

# Build executable (depends on static lib)
$(TARGET): $(OBJECTS) $(KERNEL_ENGINE_DIR)/libomnidb_opencl.a
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)
	@echo "Build complete: $(TARGET) (single executable, no .so needed)"

# Create build directories
$(BUILD_DIR)/CoProcessor:
	@mkdir -p $(BUILD_DIR)/CoProcessor

# Compile C++ source files
$(BUILD_DIR)/CoProcessor/%.o: CoProcessor/%.cpp | $(BUILD_DIR)/CoProcessor
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Build the KernelEngine static library first
kernel-engine: $(KERNEL_ENGINE_DIR)/libomnidb_opencl.a

$(KERNEL_ENGINE_DIR)/libomnidb_opencl.a:
	$(MAKE) -C $(KERNEL_ENGINE_DIR) static

# Build everything from scratch
full-build: kernel-engine all

# Deploy to bin/release/KernelScheduler
deploy: $(TARGET)
	cp $(TARGET) ../../../bin/release/KernelScheduler/CoProcessor
	@echo "Deployed to bin/release/KernelScheduler/"

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TARGET)
	@echo "Clean complete"

# Debug build
debug: CXXFLAGS += -g -DDEBUG
debug: $(TARGET)

# Help target
help:
	@echo "Available targets:"
	@echo "  all            - Build the CoProcessor executable (default)"
	@echo "  kernel-engine  - Build the KernelEngine static library"
	@echo "  full-build     - Build KernelEngine + CoProcessor"
	@echo "  deploy         - Copy binary to bin/release/KernelScheduler/"
	@echo "  clean          - Remove build artifacts"
	@echo "  debug          - Build with debug symbols"
	@echo "  help           - Show this message"
	@echo ""
	@echo "Build order:"
	@echo "  1. make kernel-engine  (or: cd $(KERNEL_ENGINE_DIR) && make static)"
	@echo "  2. make"
	@echo "  3. make deploy"
	@echo ""
	@echo "Requirements:"
	@echo "  - OpenCL development headers (ocl-icd-opencl-dev)"
	@echo "  - PoCL or Intel OpenCL runtime"
	@echo "  - libomnidb_opencl.a (built from KernelEngine)"

.PHONY: all clean deploy debug help kernel-engine full-build
