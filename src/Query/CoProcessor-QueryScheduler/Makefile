# ============================================================================
# Makefile for CoProcessor
# Builds all .cpp sources under CoProcessor/ and links with libomnidb_opencl.so
# ============================================================================

# Compiler
CXX      := g++
CXXFLAGS := -std=c++11 -fopenmp -Wall -Wno-unused-variable -Wno-unused-but-set-variable -O2

# Directories
PROJ_DIR  := .
SRC_DIR   := $(PROJ_DIR)/CoProcessor
LIB_DIR   := $(PROJ_DIR)/lib
BUILD_DIR := $(PROJ_DIR)/build

# Include paths (MyLib, TonyLib, MianLib are referenced via relative paths
# from the source files, so the parent directory must be on the include path)
INCLUDES := -I$(PROJ_DIR)

# OpenCL
# Adjust OPENCL_INC / OPENCL_LIB if your OpenCL SDK is installed elsewhere
OPENCL_INC ?=
OPENCL_LIB ?=
ifneq ($(OPENCL_INC),)
  INCLUDES += -I$(OPENCL_INC)
endif
ifneq ($(OPENCL_LIB),)
  LDFLAGS_EXTRA := -L$(OPENCL_LIB)
endif

# Linker flags
LDFLAGS  := -L$(LIB_DIR) $(LDFLAGS_EXTRA) -Wl,--allow-shlib-undefined -Wl,-rpath,'$$ORIGIN'
LDLIBS   := -lomnidb_opencl -lOpenCL -lpthread -ldl

# Target binary (output to current directory)
TARGET   := $(PROJ_DIR)/CoProcessorApp

# --------------------------------------------------------------------------
# Source / object lists
# --------------------------------------------------------------------------
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)
	@cp -u $(LIB_DIR)/libomnidb_opencl.so $(PROJ_DIR)/
	@cp -u $(SRC_DIR)/primitive.cl $(PROJ_DIR)/ 2>/dev/null || cp -u $(LIB_DIR)/primitive.cl $(PROJ_DIR)/ 2>/dev/null || true
	@test -f $(SRC_DIR)/RS.conf && cp -u $(SRC_DIR)/RS.conf $(PROJ_DIR)/ || true
	@test -f $(SRC_DIR)/dbmbench.conf && cp -u $(SRC_DIR)/dbmbench.conf $(PROJ_DIR)/ || true
	@echo ""
	@echo "=== Build complete: $(TARGET) ==="
	@echo "Run with:  ./CoProcessorApp <numQueries> <numThreads>"

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(PROJ_DIR)/CoProcessorApp $(PROJ_DIR)/libomnidb_opencl.so
