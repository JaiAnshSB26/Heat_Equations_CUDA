CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic
TARGET   := heat2d
SRCS     := main.cpp heat2d.cpp
OBJS     := $(SRCS:.cpp=.o)

# CUDA build. sm_86 = RTX 3060 (Ampere); change for other GPUs.
# reminder - run nvidia-smi to get the architecture of GPU, change sm_86 to its compute capability or drop -arch to use nvcc's default.

#Changing around stuff for the school machines.
CUDA_HOME := /usr/local/cuda-12.9.1

NVCC        := $(CUDA_HOME)/bin/nvcc
#NVCC        := nvcc
#NVCCFLAGS   := -std=c++17 -O2 -arch=sm_86   <--- My local Dell G15 Nvidia RTX 3060.
# NVCCFLAGS   := -std=c++17 -O2 -arch=sm_89    <--- School machines were a bit weird with <cuda_runtime.h> so I added in the hard gates to ensure build doesn't crash, use this if that one is too restrictive for another school machine!
NVCCFLAGS := -std=c++17 -O2 -arch=sm_89 \
             -I$(CUDA_HOME)/targets/x86_64-linux/include \
             -L$(CUDA_HOME)/targets/x86_64-linux/lib
CUDA_TARGET := heat2d_cuda
CUDA_SRCS   := main_cuda.cpp heat2d.cpp heat2d_cuda.cu

# Optional/bonus: implicit (backward-Euler, Jacobi) GPU solver.
IMPLICIT_TARGET := heat2d_implicit
IMPLICIT_SRCS   := main_implicit_cuda.cpp heat2d.cpp heat2d_implicit.cu

# Build
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp heat2d.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# CUDA build: nvcc compiles the .cu kernels and the host sources in one shot.
cuda: $(CUDA_TARGET)

$(CUDA_TARGET): $(CUDA_SRCS) heat2d.hpp heat2d_cuda.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $(CUDA_SRCS)

# Convenience targets (that's what we call it I guess?)
run: $(TARGET)
	./$(TARGET)

run_cuda: $(CUDA_TARGET)
	./$(CUDA_TARGET)

# Optional/bonus implicit solver.
implicit_cuda: $(IMPLICIT_TARGET)

$(IMPLICIT_TARGET): $(IMPLICIT_SRCS) heat2d.hpp heat2d_implicit.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $(IMPLICIT_SRCS)

run_implicit_cuda: $(IMPLICIT_TARGET)
	./$(IMPLICIT_TARGET)

# FUTURE THING I Planned - Render PNG snapshots and an animated GIF (requires gnuplot >= 5). This is for make plot method.
plot:
	gnuplot plot.gp

# Remove build artefacts; (FUTURE) keep snapshots so gnuplot can still read them.
clean:
	rm -f $(TARGET) $(OBJS) $(CUDA_TARGET) $(IMPLICIT_TARGET)

# Remove everything including generated data.
distclean: clean
	rm -rf snapshots
#(FUTUTE AGAIN).
.PHONY: run run_cuda cuda implicit_cuda run_implicit_cuda plot clean distclean
