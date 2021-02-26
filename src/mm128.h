#include <immintrin.h>
#include <cstring>
#include <iostream>

/************************************************************************************ mm128 code ************************************************************************************/
namespace bftile {
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

inline void multiplyTileEffAddr(const __m128i **amat, const __m128i * breord, __m128i **res) {
  __m128i atmp; // Temporary register for reodering A

  // We could potentially hold the whole tile in registers, since we don't require that many by statically unrolling this loop
  // The advantage of this method is that we should have extreme memory locality and cache locality. While A is indeed manipulated
  // on the fly, it is kept in registers which should make the operation crazy fast.
  // B is accessed consecutively and the whole tile could be kept into registers if we unroll the loop
  // C is accessed one register at a time and consecutively. No expensive scatter instructions

    // Multiply 0
  *res[0] = _mm_dpbusds_epi32(*res[0], *amat[0], breord[0]);
  *res[1] = _mm_dpbusds_epi32(*res[1], *amat[1], breord[0]);
  *res[2] = _mm_dpbusds_epi32(*res[2], *amat[2], breord[0]);
  *res[3] = _mm_dpbusds_epi32(*res[3], *amat[3], breord[0]);

  // Multiply 1: //Shuffle A in the same way as B was permuted and the multiply
  auto static const constexpr mask1 = _MM_SHUFFLE(2,3,0,1); // it's reversed because of being big endian
  atmp = _mm_shuffle_epi32(*amat[0], mask1);
  *res[0] = _mm_dpbusds_epi32(*res[0], atmp, breord[1]);
  atmp = _mm_shuffle_epi32(*amat[1], mask1);
  *res[1] = _mm_dpbusds_epi32(*res[1], atmp, breord[1]);
  atmp = _mm_shuffle_epi32(*amat[2], mask1);
  *res[2] = _mm_dpbusds_epi32(*res[2], atmp, breord[1]);
  atmp = _mm_shuffle_epi32(*amat[3], mask1);
  *res[3] = _mm_dpbusds_epi32(*res[3], atmp, breord[1]);

  // Multiply 2: //Shuffle A in the same way as B was permuted and the multiply
  auto static const constexpr mask2 = _MM_SHUFFLE(1,0,2,3);
  atmp = _mm_shuffle_epi32(*amat[0], mask2);
  *res[0] = _mm_dpbusds_epi32(*res[0], atmp, breord[2]);
  atmp = _mm_shuffle_epi32(*amat[1], mask2);
  *res[1] = _mm_dpbusds_epi32(*res[1], atmp, breord[2]);
  atmp = _mm_shuffle_epi32(*amat[2], mask2);
  *res[2] = _mm_dpbusds_epi32(*res[2], atmp, breord[2]);
  atmp = _mm_shuffle_epi32(*amat[3], mask2);
  *res[3] = _mm_dpbusds_epi32(*res[3], atmp, breord[2]);

  // Multiply 3: //Shuffle A in the same way as B was permuted and the multiply
  auto static const constexpr mask3 = _MM_SHUFFLE(0,1,3,2); // it's reversed because of being big endian
  atmp = _mm_shuffle_epi32(*amat[0], mask3);
  *res[0] = _mm_dpbusds_epi32(*res[0], atmp, breord[3]);
  atmp = _mm_shuffle_epi32(*amat[1], mask3);
  *res[1] = _mm_dpbusds_epi32(*res[1], atmp, breord[3]);
  atmp = _mm_shuffle_epi32(*amat[2], mask3);
  *res[2] = _mm_dpbusds_epi32(*res[2], atmp, breord[3]);
  atmp = _mm_shuffle_epi32(*amat[3], mask3);
  *res[3] = _mm_dpbusds_epi32(*res[3], atmp, breord[3]);
}

struct breadthfirst {
  // Our width is constrainted to be multiple of (sizeof(Register)) and the other sides of the matrices need to be
  // multiple of (sizeof(Register))/4
  static void prepareBMatrix(const int8_t * in, int8_t * out, size_t rowsB, size_t colsB) {
    // We traverse the width first then the depth
    static const constexpr size_t regwidth = sizeof(__m128i); // We have two types of increments: incrementing by regwidth elements
    static const constexpr size_t numregs = sizeof(__m128i)/4; // and increments by the number of registers of B that make up a single tile
    __m128i* outmat = reinterpret_cast<__m128i*>(out);
    __m128i intile[numregs];
    size_t offset = 0;
    for (size_t j = 0; j < rowsB; j += regwidth) {
      offset += j;
      for (size_t i = 0; i < colsB; i += numregs) {  // Our tile size is 64
        for (size_t t = 0; t < numregs; t++) { // Copy a subpart of the matrix onto a tile. @TODO optimise, do away with the copy
          std::memcpy(&intile[t], &in[offset], regwidth);
          offset += rowsB; // B comes in as a column major already so to go to the next column we need to += one column
        }
        prepareBtile(intile, outmat);
        outmat = outmat + numregs; // Advance the pointer of the output reorder matrix by 4x__m128i
      }
      offset = 0;
    }
  }

  __attribute__((flatten)) static void gemm(const uint8_t * A, const int8_t * B, int32_t * C, size_t rowsA, size_t width, size_t colsB) {
    /****** Important: C is assumed to be set to 0 ******/
    static const constexpr size_t regwidth = sizeof(__m128i); // We have two types of increments: incrementing by regwidth elements
    static const constexpr size_t numregs = sizeof(__m128i)/4; // and increments by the number of registers of B that make up a single tile
    const __m128i * breord = reinterpret_cast<const __m128i *>(B);
    for (size_t t = 0; t < width; t += regwidth) {
      // t is used to iterate over columns of A (A is iterated top to bottom one (sizeof(__m128i)) at a time))
      for (size_t j = 0; j < colsB; j += numregs ) {
        // Loop breadth first of B, depth first of C. We write C one column (sizeof(__m128i)) at a time
        const __m128i * breord_cur = breord + j; // tiles always come in 4 columns
        for (size_t i = 0; i < rowsA; i += numregs ) {
          // Loop over rows of A, going to use the same tile of B
          const __m128i * amat0 = reinterpret_cast<const __m128i *>(A + i*width + t);
          const __m128i * amat1 = reinterpret_cast<const __m128i *>(A + (i+1)*width + t);
          const __m128i * amat2 = reinterpret_cast<const __m128i *>(A + (i+2)*width + t);
          const __m128i * amat3 = reinterpret_cast<const __m128i *>(A + (i+3)*width + t);
          __m128i * cres0 = reinterpret_cast<__m128i *>(C + i*colsB + j);
          __m128i * cres1 = reinterpret_cast<__m128i *>(C + (i+1)*colsB + j);
          __m128i * cres2 = reinterpret_cast<__m128i *>(C + (i+2)*colsB + j);
          __m128i * cres3 = reinterpret_cast<__m128i *>(C + (i+3)*colsB + j);
          multiplyTileEff(amat0, amat1, amat2, amat3, breord_cur, 
                          cres0, cres1, cres2, cres3);
        }
      }
      breord = breord + colsB; // Our B reordered matrix goes over the colums first and rows later
    }
  }

  struct runner {
    using gemm = bftile::breadthfirst;
    using prepareB = bftile::breadthfirst;
  };

}; // struct breadthfirst

struct depthfirst {

  static void prepareBMatrix(const int8_t * in, int8_t * out, size_t rowsB, size_t colsB) {
    // We traverse the matrix depth first, 4 columns at a time
    typedef __m128i Register;
    static const constexpr size_t regwidth = sizeof(Register); // We have two types of increments: incrementing by regwidth elements
    static const constexpr size_t numregs = sizeof(Register)/4; // and increments by the number of registers of B that make up a single tile
    Register* outmat = reinterpret_cast<Register*>(out);
    Register intile[numregs];
    for (size_t i = 0; i < colsB; i += numregs ) {  // Our tile size is 64, 4*16. We read it 4 columns at a time (sizeof(__m128i)/4 = 4)
      size_t column_start = i*rowsB; // We go 4 further 4 columns to the right
      for (size_t j = 0; j < rowsB; j += regwidth) { // We go 16 rows down at a time. 16 is what fits in one register
        size_t offset = column_start + j;
        for (size_t t = 0; t < numregs; t++) { // Copy a subpart of the matrix onto a tile. @TODO optimise, do away with the copy
          std::memcpy(&intile[t], &in[offset], regwidth);
          offset += rowsB; // B comes in as a column major already so to go to the next column we need to += one column
        }
        prepareBtile(intile, outmat);
        outmat = outmat + numregs; // Advance the pointer of the output reorder matrix by 4x__m128i
      }
    }
  }

  __attribute__((flatten)) static void gemm(const uint8_t * A, const int8_t * B, int32_t * C, size_t rowsA, size_t width, size_t colsB) {
    /****** Important: C is assumed to be set to 0 ******/
    typedef __m128i Register;
    static const constexpr size_t regwidth = sizeof(Register); // We have two types of increments: incrementing by regwidth elements
    static const constexpr size_t numregs = sizeof(Register)/4; // and increments by the number of registers of B that make up a single tile
    const Register * breord = reinterpret_cast<const Register *>(B);
    // t is used to iterate over columns of A (A is left to right (sizeof(__m128i)) at a time))
    for (size_t j = 0; j < colsB; j += numregs) { // 16/4=4
      // Loop breadth first of B, depth first of C. We write C one column (sizeof(__m128i)) at a time
      for (size_t i = 0; i < rowsA; i += numregs) { // 16/4=4
        const Register *  breord_cur = breord;
        for (size_t t = 0; t < width; t += regwidth) { // Loop over the width so we only ever write to a set of 4 consecutive registers
          // Loop over rows of A, going to use the same tile of B
          const Register * amat0 = reinterpret_cast<const Register *>(A + i*width + t);
          const Register * amat1 = reinterpret_cast<const Register *>(A + (i+1)*width + t);
          const Register * amat2 = reinterpret_cast<const Register *>(A + (i+2)*width + t);
          const Register * amat3 = reinterpret_cast<const Register *>(A + (i+3)*width + t);
          Register * cres0 = reinterpret_cast<Register *>(C + i*colsB + j);
          Register * cres1 = reinterpret_cast<Register *>(C + (i+1)*colsB + j);
          Register * cres2 = reinterpret_cast<Register *>(C + (i+2)*colsB + j);
          Register * cres3 = reinterpret_cast<Register *>(C + (i+3)*colsB + j);
          multiplyTileEff(amat0, amat1, amat2, amat3, breord_cur, 
                          cres0, cres1, cres2, cres3);
          breord_cur = breord_cur + numregs; // 16/4=4
        }
      }
      breord = breord + width/numregs; // Our B reordered matrix goes over the colums first and rows later. Divided by 4 since we use 4 registers
    }
  }

  struct runner {
    using gemm = bftile::depthfirst;
    using prepareB = bftile::depthfirst;
  };

}; //struct depthfirst

struct depthfirstaddr {
  __attribute__((flatten)) static void gemm(const uint8_t * A, const int8_t * B, int32_t * C, size_t rowsA, size_t width, size_t colsB) {
    /****** Important: C is assumed to be set to 0 ******/
    typedef __m128i Register;
    static const constexpr size_t regwidth = sizeof(Register); // We have two types of increments: incrementing by regwidth elements
    static const constexpr size_t numregs = sizeof(Register)/4; // and increments by the number of registers of B that make up a single tile
    const Register * breord = reinterpret_cast<const Register *>(B);
    const Register * amat[numregs];
    Register * cres[numregs];
    // t is used to iterate over columns of A (A is left to right (sizeof(__m128i)) at a time))
    for (size_t j = 0; j < colsB; j += numregs) { // 16/4=4
      // Loop breadth first of B, depth first of C. We write C one column (sizeof(__m128i)) at a time
      for (size_t i = 0; i < rowsA; i += numregs) { // 16/4=4
        const Register *  breord_cur = breord;
        for (size_t t = 0; t < width; t += regwidth) { // Loop over the width so we only ever write to a set of 4 consecutive registers
          // Loop over rows of A, going to use the same tile of B
          amat[0] = reinterpret_cast<const Register *>(A + i*width + t);
          amat[1] = reinterpret_cast<const Register *>(A + (i+1)*width + t);
          amat[2] = reinterpret_cast<const Register *>(A + (i+2)*width + t);
          amat[3] = reinterpret_cast<const Register *>(A + (i+3)*width + t);
          cres[0] = reinterpret_cast<Register *>(C + i*colsB + j);
          cres[1] = reinterpret_cast<Register *>(C + (i+1)*colsB + j);
          cres[2] = reinterpret_cast<Register *>(C + (i+2)*colsB + j);
          cres[3] = reinterpret_cast<Register *>(C + (i+3)*colsB + j);
          multiplyTileEffAddr(amat, breord_cur, cres);
          breord_cur = breord_cur + numregs; // 16/4=4
        }
      }
      breord = breord + width/numregs; // Our B reordered matrix goes over the colums first and rows later. Divided by 4 since we use 4 registers
    }
  }
  struct runner {
    using gemm = bftile::depthfirstaddr;
    using prepareB = bftile::depthfirst;
  };
}; //struct depthfirstaddr

struct depthfirstaddrlooptileloop {

  static inline void zeroLoop(__m128i& res, const __m128i& amat, const __m128i& breord) {
    res = _mm_dpbusds_epi32(res, amat, breord);
  }

  static inline void oneLoop(__m128i& res, const __m128i& amat, const __m128i& breord, __m128i& atmp) {
    auto static const constexpr mask1 = _MM_SHUFFLE(2,3,0,1); // it's reversed because of being big endian
    atmp = _mm_shuffle_epi32(amat, mask1);
    res = _mm_dpbusds_epi32(res, atmp, breord);
  }

  static inline void twoLoop(__m128i& res, const __m128i& amat, const __m128i& breord, __m128i& atmp) {
    auto static const constexpr mask2 = _MM_SHUFFLE(1,0,2,3); // it's reversed because of being big endian
    atmp = _mm_shuffle_epi32(amat, mask2);
    res = _mm_dpbusds_epi32(res, atmp, breord);
  }

  static inline void threeLoop(__m128i& res, const __m128i& amat, const __m128i& breord, __m128i& atmp) {
    auto static const constexpr mask3 = _MM_SHUFFLE(0,1,3,2);  // it's reversed because of being big endian
    atmp = _mm_shuffle_epi32(amat, mask3);
    res = _mm_dpbusds_epi32(res, atmp, breord);
  }

  __attribute__((flatten)) static inline void multiplyTileEffAddrLoop(const __m128i **amat, const __m128i * breord, __m128i **res) {
    __m128i atmp; // Temporary register for reodering A

    // We could potentially hold the whole tile in registers, since we don't require that many by statically unrolling this loop
    // The advantage of this method is that we should have extreme memory locality and cache locality. While A is indeed manipulated
    // on the fly, it is kept in registers which should make the operation crazy fast.
    // B is accessed consecutively and the whole tile could be kept into registers if we unroll the loop
    // C is accessed one register at a time and consecutively. No expensive scatter instructions

    // Multiply 0
    for (int i = 0; i<4; i++) {
      zeroLoop(*res[i], *amat[i], breord[0]);
    }    

    // Multiply 1: //Shuffle A in the same way as B was permuted and the multiply
    for (int i = 0; i<4; i++) {
      oneLoop(*res[i], *amat[i], breord[1], atmp);
    }

    // Multiply 2: //Shuffle A in the same way as B was permuted and the multiply
    for (int i = 0; i<4; i++) {
      twoLoop(*res[i], *amat[i], breord[2], atmp);
    }

    // Multiply 3: //Shuffle A in the same way as B was permuted and the multiply
    for (int i = 0; i<4; i++) {
      threeLoop(*res[i], *amat[i], breord[3], atmp);
    }
  }

  __attribute__((flatten)) static void gemm(const uint8_t * A, const int8_t * B, int32_t * C, size_t rowsA, size_t width, size_t colsB) {
    /****** Important: C is assumed to be set to 0 ******/
    typedef __m128i Register;
    static const constexpr size_t regwidth = sizeof(Register); // We have two types of increments: incrementing by regwidth elements
    static const constexpr size_t numregs = sizeof(Register)/4; // and increments by the number of registers of B that make up a single tile
    const Register * breord = reinterpret_cast<const Register *>(B);
    const Register * amat[numregs];
    Register * cres[numregs];
    // t is used to iterate over columns of A (A is left to right (sizeof(__m128i)) at a time))
    for (size_t j = 0; j < colsB; j += numregs) { // 16/4=4
      // Loop breadth first of B, depth first of C. We write C one column (sizeof(__m128i)) at a time
      for (size_t i = 0; i < rowsA; i += numregs) { // 16/4=4
        const Register *  breord_cur = breord;
        for (size_t t = 0; t < width; t += regwidth) { // Loop over the width so we only ever write to a set of 4 consecutive registers
          // Loop over rows of A, going to use the same tile of B
          for (size_t n = 0; n < numregs; n++) { // Read in from the memory @TODO kpu unordered_unfurl? Could also use pragma unroll but it's compiler dependent...
            amat[n] = reinterpret_cast<const Register *>(A + (i+n)*width + t);
            cres[n] = reinterpret_cast<Register *>(C + (i+n)*colsB + j);
          }
          multiplyTileEffAddrLoop(amat, breord_cur, cres);
          breord_cur = breord_cur + numregs; // 16/4=4
        }
      }
      breord = breord + width/numregs; // Our B reordered matrix goes over the colums first and rows later. Divided by 4 since we use 4 registers
    }
  }
  struct runner {
    using gemm = bftile::depthfirstaddrlooptileloop;
    using prepareB = bftile::depthfirst;
  };
}; //struct depthfirstaddrlooptileloop

struct depthfirstaddrloop {
  __attribute__((flatten)) static void gemm(const uint8_t * A, const int8_t * B, int32_t * C, size_t rowsA, size_t width, size_t colsB) {
    /****** Important: C is assumed to be set to 0 ******/
    typedef __m128i Register;
    static const constexpr size_t regwidth = sizeof(Register); // We have two types of increments: incrementing by regwidth elements
    static const constexpr size_t numregs = sizeof(Register)/4; // and increments by the number of registers of B that make up a single tile
    const Register * breord = reinterpret_cast<const Register *>(B);
    const Register * amat[numregs];
    Register * cres[numregs];
    // t is used to iterate over columns of A (A is left to right (sizeof(__m128i)) at a time))
    for (size_t j = 0; j < colsB; j += numregs) { // 16/4=4
      // Loop breadth first of B, depth first of C. We write C one column (sizeof(__m128i)) at a time
      for (size_t i = 0; i < rowsA; i += numregs) { // 16/4=4
        const Register *  breord_cur = breord;
        for (size_t t = 0; t < width; t += regwidth) { // Loop over the width so we only ever write to a set of 4 consecutive registers
          // Loop over rows of A, going to use the same tile of B
          for (size_t n = 0; n < numregs; n++) { // Read in from the memory @TODO kpu unordered_unfurl? Could also use pragma unroll but it's compiler dependent...
            amat[n] = reinterpret_cast<const Register *>(A + (i+n)*width + t);
            cres[n] = reinterpret_cast<Register *>(C + (i+n)*colsB + j);
          }
          multiplyTileEffAddr(amat, breord_cur, cres);
          breord_cur = breord_cur + numregs; // 16/4=4
        }
      }
      breord = breord + width/numregs; // Our B reordered matrix goes over the colums first and rows later. Divided by 4 since we use 4 registers
    }
  }
  struct runner {
    using gemm = bftile::depthfirstaddrloop;
    using prepareB = bftile::depthfirst;
  };
}; //struct depthfirstaddrloop

struct depthfirstaddrlooptileloopwritedepend {

  static inline void multiplyTileSeqWrite(const __m128i ** amat, const __m128i * breord, __m128i ** res) {
    __m128i atmp; // Temporary register for reodering A

    // We could potentially hold the whole tile in registers, since we don't require that many by statically unrolling this loop
    // The advantage of this method is that we should have extreme memory locality and cache locality. While A is indeed manipulated
    // on the fly, it is kept in registers which should make the operation crazy fast.
    // B is accessed consecutively and the whole tile could be kept into registers if we unroll the loop
    // C is accessed one register at a time and consecutively. No expensive scatter instructions
    for (int i = 0; i < 4; i++) {
      // We load A once, perform four multiply-adds and write into C. The four dpbusds operations will produce for consecutive for-major results
      // Additional work is 3 permute operations and additional space required is one temporary register

      // Multiply 0
      *res[i] = _mm_dpbusds_epi32(*res[i], *amat[i], breord[0]);

      // Multiply 1: //Shuffle A in the same way as B was permuted and the multiply
      auto static const constexpr mask1 = _MM_SHUFFLE(2,3,0,1); // it's reversed because of being big endian
      atmp = _mm_shuffle_epi32(*amat[i], mask1);
      *res[i] = _mm_dpbusds_epi32(*res[i], atmp, breord[1]);

      // Multiply 2: //Shuffle A in the same way as B was permuted and the multiply
      auto static const constexpr mask2 = _MM_SHUFFLE(1,0,2,3);
      atmp = _mm_shuffle_epi32(*amat[i], mask2);
      *res[i] = _mm_dpbusds_epi32(*res[i], atmp, breord[2]);

      // Multiply 3: //Shuffle A in the same way as B was permuted and the multiply
      auto static const constexpr mask3 = _MM_SHUFFLE(0,1,3,2); // it's reversed because of being big endian
      atmp = _mm_shuffle_epi32(*amat[i], mask3);
      *res[i] = _mm_dpbusds_epi32(*res[i], atmp, breord[3]);
    }
  }

  __attribute__((flatten)) static void gemm(const uint8_t * A, const int8_t * B, int32_t * C, size_t rowsA, size_t width, size_t colsB) {
    /****** Important: C is assumed to be set to 0 ******/
    typedef __m128i Register;
    static const constexpr size_t regwidth = sizeof(Register); // We have two types of increments: incrementing by regwidth elements
    static const constexpr size_t numregs = sizeof(Register)/4; // and increments by the number of registers of B that make up a single tile
    const Register * breord = reinterpret_cast<const Register *>(B);
    const Register * amat[numregs];
    Register * cres[numregs];
    // t is used to iterate over columns of A (A is left to right (sizeof(__m128i)) at a time))
    for (size_t j = 0; j < colsB; j += numregs) { // 16/4=4
      // Loop breadth first of B, depth first of C. We write C one column (sizeof(__m128i)) at a time
      for (size_t i = 0; i < rowsA; i += numregs) { // 16/4=4
        const Register *  breord_cur = breord;
        for (size_t t = 0; t < width; t += regwidth) { // Loop over the width so we only ever write to a set of 4 consecutive registers
          // Loop over rows of A, going to use the same tile of B
          for (size_t n = 0; n < numregs; n++) { // Read in from the memory @TODO kpu unordered_unfurl? Could also use pragma unroll but it's compiler dependent...
            amat[n] = reinterpret_cast<const Register *>(A + (i+n)*width + t);
            cres[n] = reinterpret_cast<Register *>(C + (i+n)*colsB + j);
          }
          multiplyTileSeqWrite(amat, breord_cur, cres);
          breord_cur = breord_cur + numregs; // 16/4=4
        }
      }
      breord = breord + width/numregs; // Our B reordered matrix goes over the colums first and rows later. Divided by 4 since we use 4 registers
    }
  }
  struct runner {
    using gemm = bftile::depthfirstaddrlooptileloopwritedepend;
    using prepareB = bftile::depthfirst;
  };
}; //struct depthfirstaddrlooptileloopwritedepend

} // namespace bftile
