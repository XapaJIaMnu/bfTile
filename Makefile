demo.out: demo.cpp
	$(CXX) demo.cpp -march=native -O3 -Wall -Wextra -o demo.out -std=c++14

all: demo.out

clean:
	rm demo.out
