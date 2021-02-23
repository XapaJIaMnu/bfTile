#include <immintrin.h>
#include <cstring>
#include <iostream>

namespace bftile {
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
} // namespace bftile
