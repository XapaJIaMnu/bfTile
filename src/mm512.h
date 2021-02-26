#include <immintrin.h>
#include <cstring>
#include <iostream>

namespace bftile {
  namespace mm512 {
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

struct depthfirst {
  static void prepareBMatrix(const int8_t * in, int8_t * out, size_t rowsB, size_t colsB) {
    // We traverse the matrix depth first, 4 columns at a time
    typedef __m512i Register;
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
};

struct depthfirstaddrlooptileloopwritedepend {

  static inline void multiplyTileSeqWrite(const __m512i ** amat, const __m512i * breord, __m512i ** res) {
    __m512i atmp; // Temporary register for reodering A
    __m512i laneSwappedA;

    // We could potentially hold the whole tile in registers, since we don't require that many by statically unrolling this loop
    // The advantage of this method is that we should have extreme memory locality and cache locality. While A is indeed manipulated
    // on the fly, it is kept in registers which should make the operation crazy fast.
    // B is accessed consecutively and the whole tile could be kept into registers if we unroll the loop
    // C is accessed one register at a time and consecutively. No expensive scatter instructions
    for (int i = 0; i < 16; i++) {
      // We load A once, perform four multiply-adds and write into C. The four dpbusds operations will produce for consecutive for-major *results
      // Additional work is 3 permute operations and additional space required is one temporary register

      // Multiply 0
      *res[i] = _mm512_dpbusds_epi32 (*res[i], *amat[i], breord[0]);

      // Multiply 1: //Shuffle A in the same way as B was permuted and the multiply
      auto static const constexpr mask1 = (_MM_PERM_ENUM)_MM_SHUFFLE(2,3,0,1); // it's reversed because of being big endian
      atmp = _mm512_shuffle_epi32(*amat[i], mask1);
      *res[i] = _mm512_dpbusds_epi32(*res[i], atmp, breord[1]);

      // Multiply 2: //Shuffle A in the same way as B was permuted and the multiply
      auto static const constexpr mask2 = (_MM_PERM_ENUM)_MM_SHUFFLE(1,0,2,3);
      atmp = _mm512_shuffle_epi32(*amat[i], mask2);
      *res[i] = _mm512_dpbusds_epi32(*res[i], atmp, breord[2]);

      // Multiply 3: //Shuffle A in the same way as B was permuted and the multiply
      auto static const constexpr mask3 = (_MM_PERM_ENUM)_MM_SHUFFLE(0,1,3,2); // it's reversed because of being big endian
      atmp = _mm512_shuffle_epi32(*amat[i], mask3);
      *res[i] = _mm512_dpbusds_epi32(*res[i], atmp, breord[3]);

      //Lane swap *amat[i] here:
      laneSwappedA = _mm512_shuffle_i32x4(*amat[i], *amat[i], 0b0100'1110);

      // Multiply 4
      *res[i] = _mm512_dpbusds_epi32(*res[i], laneSwappedA, breord[4]);

      // Multiply 5: //Shuffle A in the same way as B was permuted and the multiply
      // auto static const constexpr mask1 = _MM_SHUFFLE(2,3,0,1); // it's reversed because of being big endian
      atmp = _mm512_shuffle_epi32(laneSwappedA, mask1);
      *res[i] = _mm512_dpbusds_epi32(*res[i], atmp, breord[5]);

      // Multiply 6: //Shuffle A in the same way as B was permuted and the multiply
      // auto static const constexpr mask2 = _MM_SHUFFLE(1,0,2,3);
      atmp = _mm512_shuffle_epi32(laneSwappedA, mask2);
      *res[i] = _mm512_dpbusds_epi32(*res[i], atmp, breord[6]);

      // Multiply 7: //Shuffle A in the same way as B was permuted and the multiply
      // auto static const constexpr mask3 = _MM_SHUFFLE(0,1,3,2); // it's reversed because of being big endian
      atmp = _mm512_shuffle_epi32(laneSwappedA, mask3);
      *res[i] = _mm512_dpbusds_epi32(*res[i], atmp, breord[7]);

      //Lane swap *amat[i] here:
      laneSwappedA = _mm512_shuffle_i32x4(*amat[i], *amat[i], 0b0001'1011);

      // Multiply 8
      *res[i] = _mm512_dpbusds_epi32(*res[i], laneSwappedA, breord[8]);

      // Multiply 9: //Shuffle A in the same way as B was permuted and the multiply
      // auto static const constexpr mask1 = _MM_SHUFFLE(2,3,0,1); // it's reversed because of being big endian
      atmp = _mm512_shuffle_epi32(laneSwappedA, mask1);
      *res[i] = _mm512_dpbusds_epi32 (*res[i], atmp, breord[9]);

      // Multiply 10: //Shuffle A in the same way as B was permuted and the multiply
      // auto static const constexpr mask2 = _MM_SHUFFLE(1,0,2,3);
      atmp = _mm512_shuffle_epi32(laneSwappedA, mask2);
      *res[i] = _mm512_dpbusds_epi32(*res[i], atmp, breord[10]);

      // Multiply 11: //Shuffle A in the same way as B was permuted and the multiply
      // auto static const constexpr mask3 = _MM_SHUFFLE(0,1,3,2); // it's reversed because of being big endian
      atmp = _mm512_shuffle_epi32(laneSwappedA, mask3);
      *res[i] = _mm512_dpbusds_epi32 (*res[i], atmp, breord[11]);

      //Lane swap *amat[i] here:
      laneSwappedA = _mm512_shuffle_i32x4(*amat[i], *amat[i], 0b1011'0001);

      // Multiply 12
      *res[i] = _mm512_dpbusds_epi32(*res[i], laneSwappedA, breord[12]);

      // Multiply 13: //Shuffle A in the same way as B was permuted and the multiply
      // auto static const constexpr mask1 = _MM_SHUFFLE(2,3,0,1); // it's reversed because of being big endian
      atmp = _mm512_shuffle_epi32(laneSwappedA, mask1);
      *res[i] = _mm512_dpbusds_epi32(*res[i], atmp, breord[13]);

      // Multiply 14: //Shuffle A in the same way as B was permuted and the multiply
      // auto static const constexpr mask2 = _MM_SHUFFLE(1,0,2,3);
      atmp = _mm512_shuffle_epi32(laneSwappedA, mask2);
      *res[i] = _mm512_dpbusds_epi32(*res[i], atmp, breord[14]);

      // Multiply 15: //Shuffle A in the same way as B was permuted and the multiply
      // auto static const constexpr mask3 = _MM_SHUFFLE(0,1,3,2); // it's reversed because of being big endian
      atmp = _mm512_shuffle_epi32(laneSwappedA, mask3);
      *res[i] = _mm512_dpbusds_epi32(*res[i], atmp, breord[15]);
    }
  }

  __attribute__((flatten)) static void gemm(const uint8_t * A, const int8_t * B, int32_t * C, size_t rowsA, size_t width, size_t colsB) {
    /****** Important: C is assumed to be set to 0 ******/
    typedef __m512i Register;
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
      breord = breord + 4*(width/numregs); // Our B reordered matrix goes over the colums first and rows later. Divided by 4 since we use 4 registers @TODO this is not quite right
    }
  }
  struct runner {
    using gemm = bftile::mm512::depthfirstaddrlooptileloopwritedepend;
    using prepareB = bftile::mm512::depthfirst;
  };

};

} // namespace mm512
} // namespace bftile
