CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic
TARGET   := heat2d
SRCS     := main.cpp heat2d.cpp
OBJS     := $(SRCS:.cpp=.o)

# Build
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp heat2d.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Convenience targets (that's what we call it I guess?)
run: $(TARGET)
	./$(TARGET)

# FUTURE THING I Planned - Render PNG snapshots and an animated GIF (requires gnuplot >= 5). This is for make plot method.
plot:
	gnuplot plot.gp

# Remove build artefacts; (FUTURE) keep snapshots so gnuplot can still read them.
clean:
	rm -f $(TARGET) $(OBJS)

# Remove everything including generated data.
distclean: clean
	rm -rf snapshots
#(FUTUTE AGAIN).
.PHONY: run plot clean distclean