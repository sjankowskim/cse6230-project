SRCS = general_purpose.cpp
OBJS = $(SRCS:.cpp=.o)

CXX = g++
CXXFLAGS = -g -Wall -pedantic -O2 -std=c++17

.PHONY: all test_code clean

all: test_code

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

test_code: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean: 
	-rm -f sim $(OBJS)