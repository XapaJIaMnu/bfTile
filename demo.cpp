#include <immintrin.h>
#include <iostream>
#include <stdlib.h>
#include <cstring>

template<class intType>
void printMat(intType * a, size_t rows, size_t cols, std::string name) {
  std::cerr << name << std::endl;
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      std::cerr << (int)a[i*cols + j] << " ";
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

void prepareBtile(__m128i *bmat, __m128i *breord) {
  //Assuming at the moment that we are dealing with 4x16 X 16x gemm, as this is just a demo
  //Assume B comes in columnMajor format

  // Column 0 is just BLEND
  breord[0] = _mm_mask_blend_epi8 (0xf0, bmat[0], bmat[1]);
  breord[0] = _mm_mask_blend_epi8 (0xf00, breord[0], bmat[2]);
  breord[0] = _mm_mask_blend_epi8 (0xf000, breord[0], bmat[3]);

  // Column 1 is BLEND + shuffle. We use int32 shuffle as we conveniently need to shuffle int8s in groups of four and 4int8s are the same width as a int32t
  breord[1] = _mm_mask_blend_epi8 (0xf0, bmat[1], bmat[0]);
  breord[1] = _mm_mask_blend_epi8 (0xf00, breord[1], bmat[3]);
  breord[1] = _mm_mask_blend_epi8 (0xf000, breord[1], bmat[2]);

  breord[1] = _mm_shuffle_epi32(breord[1], 0xb1);

  // Column 2 again BLEND + shuffle

  breord[2] = _mm_mask_blend_epi8 (0xf0, bmat[2], bmat[3]);
  breord[2] = _mm_mask_blend_epi8 (0xf00, breord[2], bmat[1]);
  breord[2] = _mm_mask_blend_epi8 (0xf000, breord[2], bmat[0]);

  breord[2] = _mm_shuffle_epi32(breord[2], 0x4b); //Correct (x, y, z, t) -> (t, z, x, y)

  // Column 3 final BLEND + shuffle

  breord[3] = _mm_mask_blend_epi8 (0xf0, bmat[3], bmat[2]);
  breord[3] = _mm_mask_blend_epi8 (0xf00, breord[3], bmat[0]);
  breord[3] = _mm_mask_blend_epi8 (0xf000, breord[3], bmat[1]);

  breord[3] = _mm_shuffle_epi32(breord[3], 0x1e); //Correct (x,y,z,t) -> (z, t, y, x)
}

void multiplyTile(__m128i * amat, __m128i * breord, __m128i * res) {
  __m128i atmp; // Temporary register for reodering A

  // We could potentially hold the whole tile in registers, since we don't require that many by statically unrolling this loop
  // The advantage of this method is that we should have extreme memory locality and cache locality. While A is indeed manipulated
  // on the fly, it is kept in registers which should make the operation crazy fast.
  // B is accessed consecutively and the whole tile could be kept into registers if we unroll the loop
  // C is accessed one register at a time and consecutively. No expensive scatter instructions
  for (int i = 0; i < 4; i++) {
    // We load A once, perform four multiply-adds and write into C. The four dpbusds operations will produce for consecutive for-major results
    // Additional work is 3 permute operations and additional space required is one temporary register
    res[i] = _mm_set1_epi32(0); // Set the initial result to zero. Potentially, do some bias reading here, or this could be done outside

    // Multiply 0
    res[i] = _mm_dpbusds_epi32 (res[i], amat[i], breord[0]);

    // Multiply 1: //Shuffle A in the same way as B was permuted and the multiply
    atmp = _mm_shuffle_epi32(amat[i], 0xb1);
    res[i] = _mm_dpbusds_epi32 (res[i], atmp, breord[1]);

    // Multiply 2: //Shuffle A in the same way as B was permuted and the multiply
    atmp = _mm_shuffle_epi32(amat[i], 0x4b);
    res[i] = _mm_dpbusds_epi32 (res[i], atmp, breord[2]);

    // Multiply 3: //Shuffle A in the same way as B was permuted and the multiply
    atmp = _mm_shuffle_epi32(amat[i], 0x1e);
    res[i] = _mm_dpbusds_epi32 (res[i], atmp, breord[3]);
  }
}

int main() {
  // Let's get some _mm example going
  __m128i * amat = reinterpret_cast<__m128i*>(aligned_alloc(16*4*sizeof(int8_t), 1024));
  __m128i * bmat = reinterpret_cast<__m128i*>(aligned_alloc(16*4*sizeof(int8_t), 1024));
  __m128i * cres = reinterpret_cast<__m128i*>(aligned_alloc(4*4*sizeof(int32_t), 1024));
  __m128i * breord = reinterpret_cast<__m128i*>(aligned_alloc(16*4*sizeof(int8_t), 1024));

  // Populate
  for (int i = 0; i<64; i++) {
    reinterpret_cast<int8_t*>(amat)[i] = (int8_t)i;
    reinterpret_cast<int8_t*>(bmat)[i] = (int8_t)i;
  }

  // We need cres to be zero. We can do this inside the hotloop out beforehand or do some bias preprocessing beforehand.
  // std::memset(cres, 0, 4*4*sizeof(int32_t));

  // This is just to print the two matrices so that we see what we are multiplying
  __m128i * bmatcolm = reinterpret_cast<__m128i*>(aligned_alloc(16*4*sizeof(int8_t), 1024));
  toColMajor(reinterpret_cast<int8_t *>(bmat), reinterpret_cast<int8_t *>(bmatcolm), 4, 16);
  // Our matrices to multiply rowM *colM
  printMat(reinterpret_cast<int8_t *>(amat), 4, 16, "A int8");
  printMat(reinterpret_cast<int8_t *>(bmatcolm), 16, 4, "B int8 colM");

  //Sanity check
  gemmRowMColM(reinterpret_cast<int8_t *>(amat), reinterpret_cast<int8_t *>(bmat), 4, 16, 4, reinterpret_cast<int *>(cres));
  printMat(reinterpret_cast<int *>(cres), 4, 4, "A * BcolM SlowMult");

  // FUll pipeline test
  prepareBtile(bmat, breord);
  multiplyTile(amat, breord, cres);
  printMat(reinterpret_cast<int32_t *>(&cres[0]), 4, 4, "A * Breord efficient");
}
