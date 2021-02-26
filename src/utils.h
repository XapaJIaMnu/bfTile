#pragma once
#include <iostream>
#include <stdlib.h>
#include <math.h>

/************************************************************************************ util ************************************************************************************/
namespace bftile {
template <class T>
int numDigits(T number) {
    int digits = 0;
    if (number <= 0) {
      digits = 1; // count the minus and take care of the zero case
    }
    while (number) {
        number /= 10;
        digits++;
    }
    return digits;
}

template<class intType>
void printMat(intType * a, size_t rows, size_t cols, std::string name, int digits = 0) {
  std::cerr << name << std::endl;
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      int numbah = (int)a[i*cols + j];
      // Pad for nice printing
      int mydigits = digits - numDigits(numbah);
      for (int t = 0; t < mydigits; t++) {
        std::cerr << ' ';
      }
      std::cerr << numbah << " ";
    }
      std::cerr << std::endl;
  }
  std::cerr << std::endl;
}

template<class intType>
void toColMajor(intType *in, intType * out, size_t rows, size_t cols) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out[j*rows + i] = in[i*cols + j];
    }
  }
}

template<class intType>
void gemmRowMColM(intType *A, intType * B, size_t rowsA, size_t width, size_t colsB, int * C) {
  for (size_t i = 0; i<rowsA; i++) {
    for (size_t j = 0; j< colsB; j++) {
      for (size_t w = 0; w<width; w++) {
        C[i*colsB + j] += A[i*width + w]*B[j*width +w];
      }
    }
  }
}

template<class intType1, class intType2>
void gemmRowMColM(intType1 *A, intType2 * B, size_t rowsA, size_t width, size_t colsB, int * C) {
  for (size_t i = 0; i<rowsA; i++) {
    for (size_t j = 0; j< colsB; j++) {
      for (size_t w = 0; w<width; w++) {
        C[i*colsB + j] += A[i*width + w]*B[j*width +w];
      }
    }
  }
}

struct matrix {
  size_t aRows;
  size_t width;
  size_t bCols;
};
} // namespace bftile
