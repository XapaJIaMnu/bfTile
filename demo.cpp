#include <immintrin.h>
#include <iostream>
#include <stdlib.h>
#include <cstring>
#include <math.h>
#include "aligned.h"
/************************************************************************************ util ************************************************************************************/

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

/************************************************************************************ mm128 code ************************************************************************************/
void prepareBtile(__m128i *bmat, __m128i *breord) {
  //Assuming at the moment that we are dealing with 4x16 X 16x gemm, as this is just a demo
  //Assume B comes in columnMajor format

  // Column 0 is just BLEND/ We use int32 blend, as we only are intersted in groups of 4
  breord[0] = _mm_blend_epi32(bmat[0],   bmat[1], 0b0010);
  breord[0] = _mm_blend_epi32(breord[0], bmat[2], 0b0100);
  breord[0] = _mm_blend_epi32(breord[0], bmat[3], 0b1000);

  // Column 1 is BLEND + shuffle.
  breord[1] = _mm_blend_epi32(bmat[1],   bmat[0], 0b0010);
  breord[1] = _mm_blend_epi32(breord[1], bmat[3], 0b0100);
  breord[1] = _mm_blend_epi32(breord[1], bmat[2], 0b1000);

  auto static const constexpr mask1 = _MM_SHUFFLE(2,3,0,1); // it's reversed because of being big endian
  breord[1] = _mm_shuffle_epi32(breord[1], mask1);  // We use int32 shuffle as we conveniently need to shuffle int8s in groups of four and 4int8s are the same width as a int32t

  // Column 2 again BLEND + shuffle
  breord[2] = _mm_blend_epi32(bmat[2],   bmat[3], 0b0010);
  breord[2] = _mm_blend_epi32(breord[2], bmat[1], 0b0100);
  breord[2] = _mm_blend_epi32(breord[2], bmat[0], 0b1000);

  auto static const constexpr mask2 = _MM_SHUFFLE(1,0,2,3); // it's reversed because of being big endian
  breord[2] = _mm_shuffle_epi32(breord[2], mask2);

  // Column 3 final BLEND + shuffle
  breord[3] = _mm_blend_epi32(bmat[3],   bmat[2], 0b0010);
  breord[3] = _mm_blend_epi32(breord[3], bmat[0], 0b0100);
  breord[3] = _mm_blend_epi32(breord[3], bmat[1], 0b1000);
  auto static const constexpr mask3 = _MM_SHUFFLE(0,1,3,2); // it's reversed because of being big endian
  breord[3] = _mm_shuffle_epi32(breord[3], mask3);
}

void prepareBMatrix(const int8_t * in, int8_t * out, size_t rowsB, size_t colsB) {
  // We traverse the width first then the depth
  __m128i* outmat = reinterpret_cast<__m128i*>(out);
  __m128i intile[4];
  size_t offset = 0;
  for (size_t j = 0; j < rowsB; j += 16) {
    offset += j;
    for (size_t i = 0; i < colsB; i += 4) {  // Our tile size is 64
      for (size_t t = 0; t < 4; t++) { // Copy a subpart of the matrix onto a tile. @TODO optimise, do away with the copy
        std::memcpy(&intile[t], &in[offset], sizeof(__m128i));
        offset += rowsB; // B comes in as a column major already so to go to the next column we need to += one column
      }
      prepareBtile(intile, outmat);
      outmat = outmat + 4; // Advance the pointer of the output reorder matrix by 4x__m128i
    }
    offset = 0;
  }
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
    res[i] = _mm_dpbusds_epi32(res[i], amat[i], breord[0]);

    // Multiply 1: //Shuffle A in the same way as B was permuted and the multiply
    auto static const constexpr mask1 = _MM_SHUFFLE(2,3,0,1); // it's reversed because of being big endian
    atmp = _mm_shuffle_epi32(amat[i], mask1);
    res[i] = _mm_dpbusds_epi32(res[i], atmp, breord[1]);

    // Multiply 2: //Shuffle A in the same way as B was permuted and the multiply
    auto static const constexpr mask2 = _MM_SHUFFLE(1,0,2,3);
    atmp = _mm_shuffle_epi32(amat[i], mask2);
    res[i] = _mm_dpbusds_epi32(res[i], atmp, breord[2]);

    // Multiply 3: //Shuffle A in the same way as B was permuted and the multiply
    auto static const constexpr mask3 = _MM_SHUFFLE(0,1,3,2); // it's reversed because of being big endian
    atmp = _mm_shuffle_epi32(amat[i], mask3);
    res[i] = _mm_dpbusds_epi32(res[i], atmp, breord[3]);
  }
}

/* The above function doesn't take into account the latency of dpbusds, which is most definitely more than one cycle.
   We need to reorder the operations in such a manner that there are no consecutive write dependecies*/
void multiplyTileEff(__m128i * amat, __m128i * breord, __m128i * res) {
  __m128i atmp; // Temporary register for reodering A

  // We could potentially hold the whole tile in registers, since we don't require that many by statically unrolling this loop
  // The advantage of this method is that we should have extreme memory locality and cache locality. While A is indeed manipulated
  // on the fly, it is kept in registers which should make the operation crazy fast.
  // B is accessed consecutively and the whole tile could be kept into registers if we unroll the loop
  // C is accessed one register at a time and consecutively. No expensive scatter instructions
  // Set 0
  for (int i = 0; i < 4; i++) {
    res[i] = _mm_set1_epi32(0);
  }

  {
    // Multiply 0
    res[0] = _mm_dpbusds_epi32(res[0], amat[0], breord[0]);
    res[1] = _mm_dpbusds_epi32(res[1], amat[1], breord[0]);
    res[2] = _mm_dpbusds_epi32(res[2], amat[2], breord[0]);
    res[3] = _mm_dpbusds_epi32(res[3], amat[3], breord[0]);

    // Multiply 1: //Shuffle A in the same way as B was permuted and the multiply
    auto static const constexpr mask1 = _MM_SHUFFLE(2,3,0,1); // it's reversed because of being big endian
    atmp = _mm_shuffle_epi32(amat[0], mask1);
    res[0] = _mm_dpbusds_epi32(res[0], atmp, breord[1]);
    atmp = _mm_shuffle_epi32(amat[1], mask1);
    res[1] = _mm_dpbusds_epi32(res[1], atmp, breord[1]);
    atmp = _mm_shuffle_epi32(amat[2], mask1);
    res[2] = _mm_dpbusds_epi32(res[2], atmp, breord[1]);
    atmp = _mm_shuffle_epi32(amat[3], mask1);
    res[3] = _mm_dpbusds_epi32(res[3], atmp, breord[1]);

    // Multiply 2: //Shuffle A in the same way as B was permuted and the multiply
    auto static const constexpr mask2 = _MM_SHUFFLE(1,0,2,3);
    atmp = _mm_shuffle_epi32(amat[0], mask2);
    res[0] = _mm_dpbusds_epi32(res[0], atmp, breord[2]);
    atmp = _mm_shuffle_epi32(amat[1], mask2);
    res[1] = _mm_dpbusds_epi32(res[1], atmp, breord[2]);
    atmp = _mm_shuffle_epi32(amat[2], mask2);
    res[2] = _mm_dpbusds_epi32(res[2], atmp, breord[2]);
    atmp = _mm_shuffle_epi32(amat[3], mask2);
    res[3] = _mm_dpbusds_epi32(res[3], atmp, breord[2]);

    // Multiply 3: //Shuffle A in the same way as B was permuted and the multiply
    auto static const constexpr mask3 = _MM_SHUFFLE(0,1,3,2); // it's reversed because of being big endian
    atmp = _mm_shuffle_epi32(amat[0], mask3);
    res[0] = _mm_dpbusds_epi32(res[0], atmp, breord[3]);
    atmp = _mm_shuffle_epi32(amat[1], mask3);
    res[1] = _mm_dpbusds_epi32(res[1], atmp, breord[3]);
    atmp = _mm_shuffle_epi32(amat[2], mask3);
    res[2] = _mm_dpbusds_epi32(res[2], atmp, breord[3]);
    atmp = _mm_shuffle_epi32(amat[3], mask3);
    res[3] = _mm_dpbusds_epi32(res[3], atmp, breord[3]);
  }
}

/* The above function doesn't take into account the latency of dpbusds, which is most definitely more than one cycle.
   We need to reorder the operations in such a manner that there are no consecutive write dependecies*/
inline void multiplyTileEff(const __m128i *amat0, const __m128i *amat1, const __m128i *amat2, const __m128i *amat3, const __m128i * breord, 
                     __m128i *res0, __m128i *res1, __m128i *res2, __m128i *res3) {
  __m128i atmp; // Temporary register for reodering A

  // We could potentially hold the whole tile in registers, since we don't require that many by statically unrolling this loop
  // The advantage of this method is that we should have extreme memory locality and cache locality. While A is indeed manipulated
  // on the fly, it is kept in registers which should make the operation crazy fast.
  // B is accessed consecutively and the whole tile could be kept into registers if we unroll the loop
  // C is accessed one register at a time and consecutively. No expensive scatter instructions

  // Multiply 0
  *res0 = _mm_dpbusds_epi32(*res0, *amat0, breord[0]);
  *res1 = _mm_dpbusds_epi32(*res1, *amat1, breord[0]);
  *res2 = _mm_dpbusds_epi32(*res2, *amat2, breord[0]);
  *res3 = _mm_dpbusds_epi32(*res3, *amat3, breord[0]);

  // Multiply 1: //Shuffle A in the same way as B was permuted and the multiply
  auto static const constexpr mask1 = _MM_SHUFFLE(2,3,0,1); // it's reversed because of being big endian
  atmp = _mm_shuffle_epi32(*amat0, mask1);
  *res0 = _mm_dpbusds_epi32(*res0, atmp, breord[1]);
  atmp = _mm_shuffle_epi32(*amat1, mask1);
  *res1 = _mm_dpbusds_epi32(*res1, atmp, breord[1]);
  atmp = _mm_shuffle_epi32(*amat2, mask1);
  *res2 = _mm_dpbusds_epi32(*res2, atmp, breord[1]);
  atmp = _mm_shuffle_epi32(*amat3, mask1);
  *res3 = _mm_dpbusds_epi32(*res3, atmp, breord[1]);

  // Multiply 2: //Shuffle A in the same way as B was permuted and the multiply
  auto static const constexpr mask2 = _MM_SHUFFLE(1,0,2,3);
  atmp = _mm_shuffle_epi32(*amat0, mask2);
  *res0 = _mm_dpbusds_epi32(*res0, atmp, breord[2]);
  atmp = _mm_shuffle_epi32(*amat1, mask2);
  *res1 = _mm_dpbusds_epi32(*res1, atmp, breord[2]);
  atmp = _mm_shuffle_epi32(*amat2, mask2);
  *res2 = _mm_dpbusds_epi32(*res2, atmp, breord[2]);
  atmp = _mm_shuffle_epi32(*amat3, mask2);
  *res3 = _mm_dpbusds_epi32(*res3, atmp, breord[2]);

  // Multiply 3: //Shuffle A in the same way as B was permuted and the multiply
  auto static const constexpr mask3 = _MM_SHUFFLE(0,1,3,2); // it's reversed because of being big endian
  atmp = _mm_shuffle_epi32(*amat0, mask3);
  *res0 = _mm_dpbusds_epi32(*res0, atmp, breord[3]);
  atmp = _mm_shuffle_epi32(*amat1, mask3);
  *res1 = _mm_dpbusds_epi32(*res1, atmp, breord[3]);
  atmp = _mm_shuffle_epi32(*amat2, mask3);
  *res2 = _mm_dpbusds_epi32(*res2, atmp, breord[3]);
  atmp = _mm_shuffle_epi32(*amat3, mask3);
  *res3 = _mm_dpbusds_epi32(*res3, atmp, breord[3]);
}

void gemm(const uint8_t * A, const int8_t * B, int32_t * C, size_t rowsA, size_t width, size_t colsB) {
  /****** Important: C is assumed to be set to 0 ******/
  size_t offsetA = 0; //Offset of A, as we are reading first 4 elements from each row. This is for when width > tile size
  size_t offsetC = 0;
  const __m128i * breord = reinterpret_cast<const __m128i *>(B);
  for (size_t t = 0; t < width; t += 16) {
    offsetA += t;
    offsetC = 0; // Reset every time we go to a new width sub-section
    for (size_t j = 0; j < colsB; j += 4) { //We do 4 columns of B at a time
      // Loop over rows of A, going to use the same tile of B
      const __m128i * breord_cur = breord + j; // tiles always come in 4 columns
      offsetC += j; //Offset of C, as we are writing 4 elements at a time
      for (size_t i = 0; i < rowsA; i += 4) {
        const __m128i * amat0 = reinterpret_cast<const __m128i *>(A + i*width + offsetA);
        const __m128i * amat1 = reinterpret_cast<const __m128i *>(A + (i+1)*width + offsetA);
        const __m128i * amat2 = reinterpret_cast<const __m128i *>(A + (i+2)*width + offsetA);
        const __m128i * amat3 = reinterpret_cast<const __m128i *>(A + (i+3)*width + offsetA);

        __m128i * cres0 = reinterpret_cast<__m128i *>(C + i*colsB + offsetC);
        __m128i * cres1 = reinterpret_cast<__m128i *>(C + (i+1)*colsB + offsetC);
        __m128i * cres2 = reinterpret_cast<__m128i *>(C + (i+2)*colsB + offsetC);
        __m128i * cres3 = reinterpret_cast<__m128i *>(C + (i+3)*colsB + offsetC);
        multiplyTileEff(amat0, amat1, amat2, amat3, breord_cur, 
                        cres0, cres1, cres2, cres3);
      }
    }
    breord = breord + colsB; // Our B reordered matrix goes over the colums first and rows later
  }
}

/************************************************************************************ mm256 code ************************************************************************************/

inline void prepareBtileSubRoutine(__m256i *bmat, __m256i *breord) {
  //Second part of shuffling requires us to lane swap all of bmat. Call this function twice to get it

  // Column 0 is just BLEND/ We use int32 blend, as we only are intersted in groups of 4
  breord[0] = _mm256_blend_epi32(bmat[0],   bmat[1], 0b00000010);
  breord[0] = _mm256_blend_epi32(breord[0], bmat[2], 0b00000100);
  breord[0] = _mm256_blend_epi32(breord[0], bmat[3], 0b00001000);
  breord[0] = _mm256_blend_epi32(breord[0], bmat[4], 0b00010000);
  breord[0] = _mm256_blend_epi32(breord[0], bmat[5], 0b00100000);
  breord[0] = _mm256_blend_epi32(breord[0], bmat[6], 0b01000000);
  breord[0] = _mm256_blend_epi32(breord[0], bmat[7], 0b10000000);

  // Column 1 is BLEND + shuffle.
  breord[1] = _mm256_blend_epi32(bmat[1],   bmat[0], 0b00000010);
  breord[1] = _mm256_blend_epi32(breord[1], bmat[3], 0b00000100);
  breord[1] = _mm256_blend_epi32(breord[1], bmat[2], 0b00001000);
  breord[1] = _mm256_blend_epi32(breord[1], bmat[5], 0b00010000);
  breord[1] = _mm256_blend_epi32(breord[1], bmat[4], 0b00100000);
  breord[1] = _mm256_blend_epi32(breord[1], bmat[7], 0b01000000);
  breord[1] = _mm256_blend_epi32(breord[1], bmat[6], 0b10000000);

  auto static const constexpr mask1 = _MM_SHUFFLE(2,3,0,1); // it's reversed because of being big endian
  breord[1] = _mm256_shuffle_epi32(breord[1], mask1);  // We use int32 shuffle as we conveniently need to shuffle int8s in groups of four and 4int8s are the same width as a int32t

  // Column 2 again BLEND + shuffle
  breord[2] = _mm256_blend_epi32(bmat[2],   bmat[3], 0b00000010);
  breord[2] = _mm256_blend_epi32(breord[2], bmat[1], 0b00000100);
  breord[2] = _mm256_blend_epi32(breord[2], bmat[0], 0b00001000);
  breord[2] = _mm256_blend_epi32(breord[2], bmat[6], 0b00010000);
  breord[2] = _mm256_blend_epi32(breord[2], bmat[7], 0b00100000);
  breord[2] = _mm256_blend_epi32(breord[2], bmat[5], 0b01000000);
  breord[2] = _mm256_blend_epi32(breord[2], bmat[4], 0b10000000);

  auto static const constexpr mask2 = _MM_SHUFFLE(1,0,2,3); // it's reversed because of being big endian
  breord[2] = _mm256_shuffle_epi32(breord[2], mask2);

  // Column 3 final BLEND + shuffle
  breord[3] = _mm256_blend_epi32(bmat[3],   bmat[2], 0b00000010);
  breord[3] = _mm256_blend_epi32(breord[3], bmat[0], 0b00000100);
  breord[3] = _mm256_blend_epi32(breord[3], bmat[1], 0b00001000);
  breord[3] = _mm256_blend_epi32(breord[3], bmat[7], 0b00010000);
  breord[3] = _mm256_blend_epi32(breord[3], bmat[6], 0b00100000);
  breord[3] = _mm256_blend_epi32(breord[3], bmat[4], 0b01000000);
  breord[3] = _mm256_blend_epi32(breord[3], bmat[5], 0b10000000);
  auto static const constexpr mask3 = _MM_SHUFFLE(0,1,3,2); // it's reversed because of being big endian
  breord[3] = _mm256_shuffle_epi32(breord[3], mask3);
}

inline void swapLanes(__m256i *input, __m256i *output) {
  for (int i = 0; i < 8; i++) {
    output[i] = _mm256_permute2x128_si256(input[i], input[i], 0b0101); // Technically we use only one input but this seems to work
  }
}

void prepareBtile(__m256i *bmat, __m256i *breord) {
  // Split into two parts that do identical things, except with lane swapped bmat
  prepareBtileSubRoutine(bmat, breord);

  //Second part of shuffling requires us to lane swap all of bmat
  __m256i * bmatlaneswap = reinterpret_cast<__m256i*>(aligned_alloc(32*8*sizeof(int8_t), 1024));
  std::memcpy(bmatlaneswap, bmat, 32*8*sizeof(int8_t));
  swapLanes(bmat, bmatlaneswap);
  prepareBtileSubRoutine(bmatlaneswap, &breord[4]);

  delete bmatlaneswap;
}


void multiplyTile(__m256i * amat, __m256i * breord, __m256i * res) {
  __m256i atmp; // Temporary register for reodering A
  __m256i laneSwappedA;

  // We could potentially hold the whole tile in registers, since we don't require that many by statically unrolling this loop
  // The advantage of this method is that we should have extreme memory locality and cache locality. While A is indeed manipulated
  // on the fly, it is kept in registers which should make the operation crazy fast.
  // B is accessed consecutively and the whole tile could be kept into registers if we unroll the loop
  // C is accessed one register at a time and consecutively. No expensive scatter instructions
  for (int i = 0; i < 8; i++) {
    // We load A once, perform four multiply-adds and write into C. The four dpbusds operations will produce for consecutive for-major results
    // Additional work is 3 permute operations and additional space required is one temporary register
    res[i] = _mm256_set1_epi32(0); // Set the initial result to zero. Potentially, do some bias reading here, or this could be done outside

    // Multiply 0
    res[i] = _mm256_dpbusds_epi32(res[i], amat[i], breord[0]);

    // Multiply 1: //Shuffle A in the same way as B was permuted and the multiply
    auto static const constexpr mask1 = _MM_SHUFFLE(2,3,0,1); // it's reversed because of being big endian
    atmp = _mm256_shuffle_epi32(amat[i], mask1);
    res[i] = _mm256_dpbusds_epi32(res[i], atmp, breord[1]);

    // Multiply 2: //Shuffle A in the same way as B was permuted and the multiply
    auto static const constexpr mask2 = _MM_SHUFFLE(1,0,2,3);
    atmp = _mm256_shuffle_epi32(amat[i], mask2);
    res[i] = _mm256_dpbusds_epi32(res[i], atmp, breord[2]);

    // Multiply 3: //Shuffle A in the same way as B was permuted and the multiply
    auto static const constexpr mask3 = _MM_SHUFFLE(0,1,3,2); // it's reversed because of being big endian
    atmp = _mm256_shuffle_epi32(amat[i], mask3);
    res[i] = _mm256_dpbusds_epi32(res[i], atmp, breord[3]);

    //Lane swap amat[i] here:
    laneSwappedA = _mm256_permute2x128_si256(amat[i], amat[i], 0b0101);

    // Multiply 4
    res[i] = _mm256_dpbusds_epi32(res[i], laneSwappedA, breord[4]);

    // Multiply 5: //Shuffle A in the same way as B was permuted and the multiply
    // auto static const constexpr mask1 = _MM_SHUFFLE(2,3,0,1); // it's reversed because of being big endian
    atmp = _mm256_shuffle_epi32(laneSwappedA, mask1);
    res[i] = _mm256_dpbusds_epi32(res[i], atmp, breord[5]);

    // Multiply 6: //Shuffle A in the same way as B was permuted and the multiply
    // auto static const constexpr mask2 = _MM_SHUFFLE(1,0,2,3);
    atmp = _mm256_shuffle_epi32(laneSwappedA, mask2);
    res[i] = _mm256_dpbusds_epi32(res[i], atmp, breord[6]);

    // Multiply 7: //Shuffle A in the same way as B was permuted and the multiply
    // auto static const constexpr mask3 = _MM_SHUFFLE(0,1,3,2); // it's reversed because of being big endian
    atmp = _mm256_shuffle_epi32(laneSwappedA, mask3);
    res[i] = _mm256_dpbusds_epi32(res[i], atmp, breord[7]);
  }
}

/************************************************************************************ mm512 code ************************************************************************************/

inline void prepareBtileSubRoutine(__m512i *bmat, __m512i *breord) {
  //Second part of shuffling requires us to lane swap all of bmat. Call this function twice to get it

  // Column 0 is just BLEND/ We use int32 blend, as we only are intersted in groups of 4. Annoyingly the instruction name and interface is slightly different from the 128 and 256 case
  breord[0] = _mm512_mask_blend_epi32(0b00000000'00000010, bmat[0],   bmat[1]);
  breord[0] = _mm512_mask_blend_epi32(0b00000000'00000100, breord[0], bmat[2]);
  breord[0] = _mm512_mask_blend_epi32(0b00000000'00001000, breord[0], bmat[3]);
  breord[0] = _mm512_mask_blend_epi32(0b00000000'00010000, breord[0], bmat[4]);
  breord[0] = _mm512_mask_blend_epi32(0b00000000'00100000, breord[0], bmat[5]);
  breord[0] = _mm512_mask_blend_epi32(0b00000000'01000000, breord[0], bmat[6]);
  breord[0] = _mm512_mask_blend_epi32(0b00000000'10000000, breord[0], bmat[7]);
  breord[0] = _mm512_mask_blend_epi32(0b00000001'00000000, breord[0], bmat[8]);
  breord[0] = _mm512_mask_blend_epi32(0b00000010'00000000, breord[0], bmat[9]);
  breord[0] = _mm512_mask_blend_epi32(0b00000100'00000000, breord[0], bmat[10]);
  breord[0] = _mm512_mask_blend_epi32(0b00001000'00000000, breord[0], bmat[11]);
  breord[0] = _mm512_mask_blend_epi32(0b00010000'00000000, breord[0], bmat[12]);
  breord[0] = _mm512_mask_blend_epi32(0b00100000'00000000, breord[0], bmat[13]);
  breord[0] = _mm512_mask_blend_epi32(0b01000000'00000000, breord[0], bmat[14]);
  breord[0] = _mm512_mask_blend_epi32(0b10000000'00000000, breord[0], bmat[15]);


  // Column 1 is BLEND + shuffle.
  breord[1] = _mm512_mask_blend_epi32(0b00000000'00000010, bmat[1],   bmat[0]);
  breord[1] = _mm512_mask_blend_epi32(0b00000000'00000100, breord[1], bmat[3]);
  breord[1] = _mm512_mask_blend_epi32(0b00000000'00001000, breord[1], bmat[2]);
  breord[1] = _mm512_mask_blend_epi32(0b00000000'00010000, breord[1], bmat[5]);
  breord[1] = _mm512_mask_blend_epi32(0b00000000'00100000, breord[1], bmat[4]);
  breord[1] = _mm512_mask_blend_epi32(0b00000000'01000000, breord[1], bmat[7]);
  breord[1] = _mm512_mask_blend_epi32(0b00000000'10000000, breord[1], bmat[6]);
  breord[1] = _mm512_mask_blend_epi32(0b00000001'00000000, breord[1], bmat[9]);
  breord[1] = _mm512_mask_blend_epi32(0b00000010'00000000, breord[1], bmat[8]);
  breord[1] = _mm512_mask_blend_epi32(0b00000100'00000000, breord[1], bmat[11]);
  breord[1] = _mm512_mask_blend_epi32(0b00001000'00000000, breord[1], bmat[10]);
  breord[1] = _mm512_mask_blend_epi32(0b00010000'00000000, breord[1], bmat[13]);
  breord[1] = _mm512_mask_blend_epi32(0b00100000'00000000, breord[1], bmat[12]);
  breord[1] = _mm512_mask_blend_epi32(0b01000000'00000000, breord[1], bmat[15]);
  breord[1] = _mm512_mask_blend_epi32(0b10000000'00000000, breord[1], bmat[14]);

  auto static const constexpr mask1 = (_MM_PERM_ENUM)_MM_SHUFFLE(2,3,0,1); // it's reversed because of being big endian. Masks on avx512 work differently too
  breord[1] = _mm512_shuffle_epi32(breord[1], mask1);  // We use int32 shuffle as we conveniently need to shuffle int8s in groups of four and 4int8s are the same width as a int32t

  // Column 2 again BLEND + shuffle
  breord[2] = _mm512_mask_blend_epi32(0b00000000'00000010, bmat[2],   bmat[3]);
  breord[2] = _mm512_mask_blend_epi32(0b00000000'00000100, breord[2], bmat[1]);
  breord[2] = _mm512_mask_blend_epi32(0b00000000'00001000, breord[2], bmat[0]);
  breord[2] = _mm512_mask_blend_epi32(0b00000000'00010000, breord[2], bmat[6]);
  breord[2] = _mm512_mask_blend_epi32(0b00000000'00100000, breord[2], bmat[7]);
  breord[2] = _mm512_mask_blend_epi32(0b00000000'01000000, breord[2], bmat[5]);
  breord[2] = _mm512_mask_blend_epi32(0b00000000'10000000, breord[2], bmat[4]);
  breord[2] = _mm512_mask_blend_epi32(0b00000001'00000000, breord[2], bmat[10]);
  breord[2] = _mm512_mask_blend_epi32(0b00000010'00000000, breord[2], bmat[11]);
  breord[2] = _mm512_mask_blend_epi32(0b00000100'00000000, breord[2], bmat[9]);
  breord[2] = _mm512_mask_blend_epi32(0b00001000'00000000, breord[2], bmat[8]);
  breord[2] = _mm512_mask_blend_epi32(0b00010000'00000000, breord[2], bmat[14]);
  breord[2] = _mm512_mask_blend_epi32(0b00100000'00000000, breord[2], bmat[15]);
  breord[2] = _mm512_mask_blend_epi32(0b01000000'00000000, breord[2], bmat[13]);
  breord[2] = _mm512_mask_blend_epi32(0b10000000'00000000, breord[2], bmat[12]);

  auto static const constexpr mask2 = (_MM_PERM_ENUM)_MM_SHUFFLE(1,0,2,3); // it's reversed because of being big endian. Masks on avx512 work differently too
  breord[2] = _mm512_shuffle_epi32(breord[2], mask2);

  // Column 3 final BLEND + shuffle
  breord[3] = _mm512_mask_blend_epi32(0b00000000'00000010, bmat[3],   bmat[2]);
  breord[3] = _mm512_mask_blend_epi32(0b00000000'00000100, breord[3], bmat[0]);
  breord[3] = _mm512_mask_blend_epi32(0b00000000'00001000, breord[3], bmat[1]);
  breord[3] = _mm512_mask_blend_epi32(0b00000000'00010000, breord[3], bmat[7]);
  breord[3] = _mm512_mask_blend_epi32(0b00000000'00100000, breord[3], bmat[6]);
  breord[3] = _mm512_mask_blend_epi32(0b00000000'01000000, breord[3], bmat[4]);
  breord[3] = _mm512_mask_blend_epi32(0b00000000'10000000, breord[3], bmat[5]);
  breord[3] = _mm512_mask_blend_epi32(0b00000001'00000000, breord[3], bmat[11]);
  breord[3] = _mm512_mask_blend_epi32(0b00000010'00000000, breord[3], bmat[10]);
  breord[3] = _mm512_mask_blend_epi32(0b00000100'00000000, breord[3], bmat[8]);
  breord[3] = _mm512_mask_blend_epi32(0b00001000'00000000, breord[3], bmat[9]);
  breord[3] = _mm512_mask_blend_epi32(0b00010000'00000000, breord[3], bmat[15]);
  breord[3] = _mm512_mask_blend_epi32(0b00100000'00000000, breord[3], bmat[14]);
  breord[3] = _mm512_mask_blend_epi32(0b01000000'00000000, breord[3], bmat[12]);
  breord[3] = _mm512_mask_blend_epi32(0b10000000'00000000, breord[3], bmat[13]);
  auto static const constexpr mask3 = (_MM_PERM_ENUM)_MM_SHUFFLE(0,1,3,2); // it's reversed because of being big endian. Masks on avx512 work differently too
  breord[3] = _mm512_shuffle_epi32(breord[3], mask3);
}
template<int imm> // imm needs to be a compile time constant, hence templates
inline void swapLanes(__m512i *input, __m512i *output) {
  for (int i = 0; i < 16; i++) {
    output[i] = _mm512_shuffle_i32x4(input[i], input[i], imm); // Technically we use only one input but this seems to work
  }
}

void prepareBtile(__m512i *bmat, __m512i *breord) {
  // Split into two parts that do identical things, except with lane swapped bmat
  prepareBtileSubRoutine(bmat, breord);

  //Second part of shuffling requires us to lane swap all of bmat
  __m512i * bmatlaneswap = reinterpret_cast<__m512i*>(aligned_alloc(32*8*sizeof(int8_t), 1024));
  std::memcpy(bmatlaneswap, bmat, 32*8*sizeof(int8_t));


  swapLanes<0b0100'1110>(bmat, bmatlaneswap);
  prepareBtileSubRoutine(bmatlaneswap, &breord[4]);

  //Third shuffling
  swapLanes<0b0001'1011>(bmat, bmatlaneswap);
  prepareBtileSubRoutine(bmatlaneswap, &breord[8]);

  //Forth shuffling
  swapLanes<0b1011'0001>(bmat, bmatlaneswap);
  prepareBtileSubRoutine(bmatlaneswap, &breord[12]);

  delete bmatlaneswap;
}


void multiplyTile(__m512i * amat, __m512i * breord, __m512i * res) {
  __m512i atmp; // Temporary register for reodering A
  __m512i laneSwappedA;

  // We could potentially hold the whole tile in registers, since we don't require that many by statically unrolling this loop
  // The advantage of this method is that we should have extreme memory locality and cache locality. While A is indeed manipulated
  // on the fly, it is kept in registers which should make the operation crazy fast.
  // B is accessed consecutively and the whole tile could be kept into registers if we unroll the loop
  // C is accessed one register at a time and consecutively. No expensive scatter instructions
  for (int i = 0; i < 16; i++) {
    // We load A once, perform four multiply-adds and write into C. The four dpbusds operations will produce for consecutive for-major results
    // Additional work is 3 permute operations and additional space required is one temporary register
    res[i] = _mm512_set1_epi32(0); // Set the initial result to zero. Potentially, do some bias reading here, or this could be done outside

    // Multiply 0
    res[i] = _mm512_dpbusds_epi32 (res[i], amat[i], breord[0]);

    // Multiply 1: //Shuffle A in the same way as B was permuted and the multiply
    auto static const constexpr mask1 = (_MM_PERM_ENUM)_MM_SHUFFLE(2,3,0,1); // it's reversed because of being big endian
    atmp = _mm512_shuffle_epi32(amat[i], mask1);
    res[i] = _mm512_dpbusds_epi32(res[i], atmp, breord[1]);

    // Multiply 2: //Shuffle A in the same way as B was permuted and the multiply
    auto static const constexpr mask2 = (_MM_PERM_ENUM)_MM_SHUFFLE(1,0,2,3);
    atmp = _mm512_shuffle_epi32(amat[i], mask2);
    res[i] = _mm512_dpbusds_epi32(res[i], atmp, breord[2]);

    // Multiply 3: //Shuffle A in the same way as B was permuted and the multiply
    auto static const constexpr mask3 = (_MM_PERM_ENUM)_MM_SHUFFLE(0,1,3,2); // it's reversed because of being big endian
    atmp = _mm512_shuffle_epi32(amat[i], mask3);
    res[i] = _mm512_dpbusds_epi32(res[i], atmp, breord[3]);

    //Lane swap amat[i] here:
    laneSwappedA = _mm512_shuffle_i32x4(amat[i], amat[i], 0b0100'1110);

    // Multiply 4
    res[i] = _mm512_dpbusds_epi32(res[i], laneSwappedA, breord[4]);

    // Multiply 5: //Shuffle A in the same way as B was permuted and the multiply
    // auto static const constexpr mask1 = _MM_SHUFFLE(2,3,0,1); // it's reversed because of being big endian
    atmp = _mm512_shuffle_epi32(laneSwappedA, mask1);
    res[i] = _mm512_dpbusds_epi32(res[i], atmp, breord[5]);

    // Multiply 6: //Shuffle A in the same way as B was permuted and the multiply
    // auto static const constexpr mask2 = _MM_SHUFFLE(1,0,2,3);
    atmp = _mm512_shuffle_epi32(laneSwappedA, mask2);
    res[i] = _mm512_dpbusds_epi32(res[i], atmp, breord[6]);

    // Multiply 7: //Shuffle A in the same way as B was permuted and the multiply
    // auto static const constexpr mask3 = _MM_SHUFFLE(0,1,3,2); // it's reversed because of being big endian
    atmp = _mm512_shuffle_epi32(laneSwappedA, mask3);
    res[i] = _mm512_dpbusds_epi32(res[i], atmp, breord[7]);

    //Lane swap amat[i] here:
    laneSwappedA = _mm512_shuffle_i32x4(amat[i], amat[i], 0b0001'1011);

    // Multiply 8
    res[i] = _mm512_dpbusds_epi32(res[i], laneSwappedA, breord[8]);

    // Multiply 9: //Shuffle A in the same way as B was permuted and the multiply
    // auto static const constexpr mask1 = _MM_SHUFFLE(2,3,0,1); // it's reversed because of being big endian
    atmp = _mm512_shuffle_epi32(laneSwappedA, mask1);
    res[i] = _mm512_dpbusds_epi32 (res[i], atmp, breord[9]);

    // Multiply 10: //Shuffle A in the same way as B was permuted and the multiply
    // auto static const constexpr mask2 = _MM_SHUFFLE(1,0,2,3);
    atmp = _mm512_shuffle_epi32(laneSwappedA, mask2);
    res[i] = _mm512_dpbusds_epi32(res[i], atmp, breord[10]);

    // Multiply 11: //Shuffle A in the same way as B was permuted and the multiply
    // auto static const constexpr mask3 = _MM_SHUFFLE(0,1,3,2); // it's reversed because of being big endian
    atmp = _mm512_shuffle_epi32(laneSwappedA, mask3);
    res[i] = _mm512_dpbusds_epi32 (res[i], atmp, breord[11]);

    //Lane swap amat[i] here:
    laneSwappedA = _mm512_shuffle_i32x4(amat[i], amat[i], 0b1011'0001);

    // Multiply 12
    res[i] = _mm512_dpbusds_epi32(res[i], laneSwappedA, breord[12]);

    // Multiply 13: //Shuffle A in the same way as B was permuted and the multiply
    // auto static const constexpr mask1 = _MM_SHUFFLE(2,3,0,1); // it's reversed because of being big endian
    atmp = _mm512_shuffle_epi32(laneSwappedA, mask1);
    res[i] = _mm512_dpbusds_epi32(res[i], atmp, breord[13]);

    // Multiply 14: //Shuffle A in the same way as B was permuted and the multiply
    // auto static const constexpr mask2 = _MM_SHUFFLE(1,0,2,3);
    atmp = _mm512_shuffle_epi32(laneSwappedA, mask2);
    res[i] = _mm512_dpbusds_epi32(res[i], atmp, breord[14]);

    // Multiply 15: //Shuffle A in the same way as B was permuted and the multiply
    // auto static const constexpr mask3 = _MM_SHUFFLE(0,1,3,2); // it's reversed because of being big endian
    atmp = _mm512_shuffle_epi32(laneSwappedA, mask3);
    res[i] = _mm512_dpbusds_epi32(res[i], atmp, breord[15]);
  }
}

/************************************************************************************ Test code ************************************************************************************/
void mm128Example() {
  std::cerr << "mm128 example" << std::endl;
  // Let's get some _mm example going
  __m128i * amat = reinterpret_cast<__m128i*>(aligned_alloc(16*4*sizeof(int8_t), 1024));
  __m128i * bmat = reinterpret_cast<__m128i*>(aligned_alloc(16*4*sizeof(int8_t), 1024));
  __m128i * cslow = reinterpret_cast<__m128i*>(aligned_alloc(4*4*sizeof(int32_t), 1024));
  __m128i * cres = reinterpret_cast<__m128i*>(aligned_alloc(4*4*sizeof(int32_t), 1024));
  __m128i * cresEff = reinterpret_cast<__m128i*>(aligned_alloc(4*4*sizeof(int32_t), 1024));
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
  printMat(reinterpret_cast<int8_t *>(amat), 4, 16, "A int8", 2);
  printMat(reinterpret_cast<int8_t *>(bmatcolm), 16, 4, "B int8 colM", 2);

  __m128i * breordcolm = reinterpret_cast<__m128i*>(aligned_alloc(16*4*sizeof(int8_t), 1024)); // For easier visualisation

  //Sanity check
  gemmRowMColM(reinterpret_cast<int8_t *>(amat), reinterpret_cast<int8_t *>(bmat), 4, 16, 4, reinterpret_cast<int *>(cslow));

  // FUll pipeline test
  prepareBtile(bmat, breord);

  // For visualisation of the reordered format
  toColMajor(reinterpret_cast<int8_t *>(breord), reinterpret_cast<int8_t *>(breordcolm), 4, 16);
  printMat(reinterpret_cast<int8_t *>(breordcolm), 16, 4, "B int8 colMreord", 2);

  multiplyTile(amat, breord, cres);
  multiplyTileEff(amat, breord, cresEff);
  printMat(reinterpret_cast<int *>(cslow), 4, 4, "A * BcolM SlowMult", 5);
  printMat(reinterpret_cast<int32_t *>(&cres[0]), 4, 4, "A * Breord efficient", 5);

  //Compare the memory
  if (std::memcmp(cslow, cres, 4*4*sizeof(int32_t))) {
    std::cerr << "mm128 fast and slow gemm implementations differ" << std::endl;
  }
  if (std::memcmp(cslow, cresEff, 4*4*sizeof(int32_t))) {
    std::cerr << "mm128 fastEfficient and slow gemm implementations differ" << std::endl;
  }
}

void mm256Example() {
  std::cerr << "mm256 example" << std::endl;
  // Let's get some _mm example going
  __m256i * amat = reinterpret_cast<__m256i*>(aligned_alloc(32*8*sizeof(int8_t), 1024));
  __m256i * bmat = reinterpret_cast<__m256i*>(aligned_alloc(32*8*sizeof(int8_t), 1024));
  __m256i * cslow = reinterpret_cast<__m256i*>(aligned_alloc(8*8*sizeof(int32_t), 1024));
  __m256i * cres = reinterpret_cast<__m256i*>(aligned_alloc(8*8*sizeof(int32_t), 1024));
  __m256i * breord = reinterpret_cast<__m256i*>(aligned_alloc(32*8*sizeof(int8_t), 1024));

  // Populate
  for (int i = 0; i<256; i++) {
    reinterpret_cast<int8_t*>(amat)[i] = (int8_t)(i%127); // A needs to be unsigned for the UBS operation, so in order to get the same results as slow gemm, make sure it fits
    reinterpret_cast<int8_t*>(bmat)[i] = (int8_t)(i); // We have unsigned times signed so one will be 0-255 other will be -128-127
  }

  // We need cres to be zero. We can do this inside the hotloop out beforehand or do some bias preprocessing beforehand.
  // std::memset(cres, 0, 4*4*sizeof(int32_t));

  // This is just to print the two matrices so that we see what we are multiplying
  __m256i * bmatcolm = reinterpret_cast<__m256i*>(aligned_alloc(32*8*sizeof(int8_t), 1024));
  toColMajor(reinterpret_cast<int8_t *>(bmat), reinterpret_cast<int8_t *>(bmatcolm), 8, 32);
  // Our matrices to multiply rowM *colM
  printMat(reinterpret_cast<int8_t *>(amat), 8, 32, "A int8", 3);
  printMat(reinterpret_cast<int8_t *>(bmatcolm), 32, 8, "B int8 colM", 4);

  __m256i * breordcolm = reinterpret_cast<__m256i*>(aligned_alloc(32*8*sizeof(int8_t), 1024)); // For easier visualisation

  //Sanity check
  gemmRowMColM(reinterpret_cast<int8_t *>(amat), reinterpret_cast<int8_t *>(bmat), 8, 32, 8, reinterpret_cast<int *>(cslow));

  // FUll pipeline test
  prepareBtile(bmat, breord);

  // For visualisation of the reordered format
  toColMajor(reinterpret_cast<int8_t *>(breord), reinterpret_cast<int8_t *>(breordcolm), 8, 32);
  printMat(reinterpret_cast<int8_t *>(breordcolm), 32, 8, "B int8 colMreord", 4);

  multiplyTile(amat, breord, cres);
  printMat(reinterpret_cast<int *>(cslow), 8, 8, "A * BcolM SlowMult", 7);
  printMat(reinterpret_cast<int32_t *>(&cres[0]), 8, 8, "A * Breord efficient", 7);

  //Compare the memory
  if (std::memcmp(cslow, cres, 8*8*sizeof(int32_t))) {
    std::cerr << "mm256 fast and slow gemm implementations differ" << std::endl;
  }
}

void mm512Example() {
  std::cerr << "mm512 example" << std::endl;
  // Let's get some _mm example going
  __m512i * amat = reinterpret_cast<__m512i*>(aligned_alloc(64*16*sizeof(int8_t), 1024));
  __m512i * bmat = reinterpret_cast<__m512i*>(aligned_alloc(64*16*sizeof(int8_t), 1024));
  __m512i * cslow = reinterpret_cast<__m512i*>(aligned_alloc(16*16*sizeof(int32_t), 1024));
  __m512i * cres = reinterpret_cast<__m512i*>(aligned_alloc(16*16*sizeof(int32_t), 1024));
  __m512i * breord = reinterpret_cast<__m512i*>(aligned_alloc(64*16*sizeof(int8_t), 1024));

  // Populate
  for (int i = 0; i<2048; i++) {
    reinterpret_cast<int8_t*>(amat)[i] = (int8_t)(i%127); // A needs to be unsigned for the UBS operation, so in order to get the same results as slow gemm, make sure it fits
    reinterpret_cast<int8_t*>(bmat)[i] = (int8_t)(i%255); // We have unsigned times signed so one will be 0-255 other will be -128-127
  }

  // We need cres to be zero. We can do this inside the hotloop out beforehand or do some bias preprocessing beforehand.
  // std::memset(cres, 0, 4*4*sizeof(int32_t));

  // This is just to print the two matrices so that we see what we are multiplying
  __m512i * bmatcolm = reinterpret_cast<__m512i*>(aligned_alloc(64*16*sizeof(int8_t), 1024));
  toColMajor(reinterpret_cast<int8_t *>(bmat), reinterpret_cast<int8_t *>(bmatcolm), 16, 64);
  // Our matrices to multiply rowM *colM
  printMat(reinterpret_cast<int8_t *>(amat), 16, 64, "A int8", 3);
  printMat(reinterpret_cast<int8_t *>(bmatcolm), 64, 16, "B int8 colM", 4);

  __m512i * breordcolm = reinterpret_cast<__m512i*>(aligned_alloc(64*16*sizeof(int8_t), 1024)); // For easier visualisation

  //Sanity check
  gemmRowMColM(reinterpret_cast<int8_t *>(amat), reinterpret_cast<int8_t *>(bmat), 16, 64, 16, reinterpret_cast<int *>(cslow));

  // FUll pipeline test
  prepareBtile(bmat, breord);

  // For visualisation of the reordered format
  toColMajor(reinterpret_cast<int8_t *>(breord), reinterpret_cast<int8_t *>(breordcolm), 16, 64);
  printMat(reinterpret_cast<int8_t *>(breordcolm), 64, 16, "B int8 colMreord", 4);

  multiplyTile(amat, breord, cres);
  printMat(reinterpret_cast<int *>(cslow), 16, 16, "A * BcolM SlowMult", 7);
  printMat(reinterpret_cast<int32_t *>(&cres[0]), 16, 16, "A * Breord efficient", 7);

  //Compare the memory
  if (std::memcmp(cslow, cres, 16*16*sizeof(int32_t))) {
    std::cerr << "mm512 fast and slow gemm implementations differ" << std::endl;
  }
}

void mm128GEMMExample(size_t aRows = 8, size_t width = 16, size_t bCols = 4, bool toprint=false) {
  AlignedVector<uint8_t> A(aRows*width);
  AlignedVector<int8_t> B(width*bCols);
  AlignedVector<int8_t> BColM(width*bCols);
  AlignedVector<int8_t> BReord(width*bCols);
  AlignedVector<int8_t> BReordColM(width*bCols);
  AlignedVector<int32_t> Cslow(aRows*bCols);
  AlignedVector<int32_t> Cfast(aRows*bCols);

  for (size_t i = 0; i < aRows*width; i++) {
    A[i] = i % 255;
  }
  for (size_t i = 0; i < width*bCols; i++) {
    B[i] = i % 255;
  }
  for (auto&& item : Cslow) {
    item = 0;
  }
  for (auto&& item : Cfast) {
    item = 0;
  }

  // Prepare datapoints
  toColMajor(B.begin(), BColM.begin(), bCols, width); // This is just for printing

  prepareBMatrix(B.begin(), BReord.begin(), width, bCols);

  //Sanity check
  gemmRowMColM(A.begin(), B.begin(), aRows, width, bCols, Cslow.begin());

  // For visualisation of the reordered format
  toColMajor(BReord.begin(), BReordColM.begin(), bCols, width);

  gemm(A.begin(), BReord.begin(), Cfast.begin(), aRows, width, bCols);

  bool wrong = false;
  if (std::memcmp(Cfast.begin(), Cslow.begin(), aRows*bCols*sizeof(int32_t))) {
    wrong = true;
  }
  if (wrong || toprint) {
    printMat(A.begin(), aRows, width, "A int8", 3);
    printMat(BColM.begin(), width, bCols, "B int8 colM", 4);
    printMat(BReordColM.begin(), width, bCols, "B int8 colM", 4);
    printMat(Cslow.begin(), aRows, bCols, "A * BcolM SlowMult", 8);
    printMat(Cfast.begin(), aRows, bCols, "A * BcolM fast", 8);
    if (wrong) {
      std::cerr << "Actual fast and slow gemm implementations differ" << std::endl;
    }
  }
}

int main() {
  //mm128Example();
  //mm256Example();
  //mm512Example();
  mm128GEMMExample();
  mm128GEMMExample(8, 16, 8);
  mm128GEMMExample(4, 16, 4);
  mm128GEMMExample(4, 32, 4);
  mm128GEMMExample(8, 32, 8);
  mm128GEMMExample(64, 32, 32);
}
