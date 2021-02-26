#include <immintrin.h>
#include <cstring>
#include <iostream>
#include "utils.h"
namespace bftile {
namespace mm256 {
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

struct depthfirst {

  static void prepareBMatrix(const int8_t * in, int8_t * out, size_t rowsB, size_t colsB) {
    // We traverse the matrix depth first, 4 columns at a time
    typedef __m256i Register;
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
/*
  __attribute__((flatten)) static void gemm(const uint8_t * A, const int8_t * B, int32_t * C, size_t rowsA, size_t width, size_t colsB) {
   // Important: C is assumed to be set to 0 
    typedef __m256i Register;
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
          std::cerr << "depthfirst gemm for __m256i is not implemented yet." << std::endl;
          std::abort();
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
    using gemm = bftile::mm256::depthfirst;
    using prepareB = bftile::mm256::depthfirst;
  }; */

}; //struct depthfirst

struct depthfirstaddrlooptileloopwritedepend {
  static inline void multiplyTileSeqWrite(const __m256i ** amat, const __m256i * breord, __m256i ** res) {
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

    // Multiply 0
    *res[i] = _mm256_dpbusds_epi32(*res[i], *amat[i], breord[0]);

    // Multiply 1: //Shuffle A in the same way as B was permuted and the multiply
    auto static const constexpr mask1 = _MM_SHUFFLE(2,3,0,1); // it's reversed because of being big endian
    atmp = _mm256_shuffle_epi32(*amat[i], mask1);
    *res[i] = _mm256_dpbusds_epi32(*res[i], atmp, breord[1]);

    // Multiply 2: //Shuffle A in the same way as B was permuted and the multiply
    auto static const constexpr mask2 = _MM_SHUFFLE(1,0,2,3);
    atmp = _mm256_shuffle_epi32(*amat[i], mask2);
    *res[i] = _mm256_dpbusds_epi32(*res[i], atmp, breord[2]);

    // Multiply 3: //Shuffle A in the same way as B was permuted and the multiply
    auto static const constexpr mask3 = _MM_SHUFFLE(0,1,3,2); // it's reversed because of being big endian
    atmp = _mm256_shuffle_epi32(*amat[i], mask3);
    *res[i] = _mm256_dpbusds_epi32(*res[i], atmp, breord[3]);

    //Lane swap amat[i] here:
    laneSwappedA = _mm256_permute2x128_si256(*amat[i], *amat[i], 0b0101);

    // Multiply 4
    *res[i] = _mm256_dpbusds_epi32(*res[i], laneSwappedA, breord[4]);

    // Multiply 5: //Shuffle A in the same way as B was permuted and the multiply
    // auto static const constexpr mask1 = _MM_SHUFFLE(2,3,0,1); // it's reversed because of being big endian
    atmp = _mm256_shuffle_epi32(laneSwappedA, mask1);
    *res[i] = _mm256_dpbusds_epi32(*res[i], atmp, breord[5]);

    // Multiply 6: //Shuffle A in the same way as B was permuted and the multiply
    // auto static const constexpr mask2 = _MM_SHUFFLE(1,0,2,3);
    atmp = _mm256_shuffle_epi32(laneSwappedA, mask2);
    *res[i] = _mm256_dpbusds_epi32(*res[i], atmp, breord[6]);

    // Multiply 7: //Shuffle A in the same way as B was permuted and the multiply
    // auto static const constexpr mask3 = _MM_SHUFFLE(0,1,3,2); // it's reversed because of being big endian
    atmp = _mm256_shuffle_epi32(laneSwappedA, mask3);
    *res[i] = _mm256_dpbusds_epi32(*res[i], atmp, breord[7]);
  }
}

  __attribute__((flatten)) static void gemm(const uint8_t * A, const int8_t * B, int32_t * C, size_t rowsA, size_t width, size_t colsB) {
    /****** Important: C is assumed to be set to 0 ******/
    typedef __m256i Register;
    static const constexpr size_t regwidth = sizeof(Register); // We have two types of increments: incrementing by regwidth elements
    static const constexpr size_t numregs = sizeof(Register)/4; // and increments by the number of registers of B that make up a single tile
    const Register * breord = reinterpret_cast<const Register *>(B);
    const Register * amat[numregs];
    Register * cres[numregs];
    // t is used to iterate over columns of A (A is left to right (sizeof(__m128i)) at a time))
    for (size_t j = 0; j < colsB; j += numregs) { // 32/4=8
      // Loop breadth first of B, depth first of C. We write C one column (sizeof(__m128i)) at a time
      for (size_t i = 0; i < rowsA; i += numregs) { // 32/4=8
        const Register *  breord_cur = breord;
        for (size_t t = 0; t < width; t += regwidth) { // Loop over the width so we only ever write to a set of 4 consecutive registers
          // Loop over rows of A, going to use the same tile of B
          for (size_t n = 0; n < numregs; n++) { // Read in from the memory @TODO kpu unordered_unfurl? Could also use pragma unroll but it's compiler dependent...
            amat[n] = reinterpret_cast<const Register *>(A + (i+n)*width + t);
            cres[n] = reinterpret_cast<Register *>(C + (i+n)*colsB + j);
          }
          multiplyTileSeqWrite(amat, breord_cur, cres);
          breord_cur = breord_cur + numregs; // 32/4=8
        }
      }
      breord = breord + 2*(width/numregs); // Our B reordered matrix goes over the colums first and rows later. Divided by 4 since we use 4 registers
    }
  }
  struct runner {
    using gemm = bftile::mm256::depthfirstaddrlooptileloopwritedepend;
    using prepareB = bftile::mm256::depthfirst;
  };

};

} // namsapce _mm256
} // namespace bftile
