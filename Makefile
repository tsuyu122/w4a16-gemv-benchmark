CXX ?= g++
CXXFLAGS = -O3 -mavx2 -mfma -fopenmp -std=c++20 -Wall -Wextra -Iinclude
LDFLAGS = -fopenmp

SRCS = src/reference.cpp src/candidate.cpp bench/harness.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = w4a16_bench.exe

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) benchmark_result.json

.PHONY: all clean
