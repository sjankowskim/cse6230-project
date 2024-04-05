SRCS = src/general_purpose.cpp src/chatgpt.cpp
OBJS = $(SRCS:.cpp=.o)

CXX = g++
CXXFLAGS = -g -Wall -pedantic -std=c++11

.PHONY: all test_code clean

all: test_code

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

test_code: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean: 
	-rm -f sim $(OBJS)