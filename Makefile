demo.out: src/demo.cpp
	$(CXX) src/demo.cpp -march=native -O3 -Wall -Wextra -o demo.out -std=c++14 -funroll-loops

all: demo.out

clean:
	rm demo.out
