#include <chrono>
#include "aligned.h"
#include "mm128.h"
#include "mm256.h"
#include "mm512.h"
#include "utils.h"
#include "do_not_optimize.h"


/************************************************************************************ Test code ************************************************************************************/
bool mm128Example(bool toprint=false) {
  using namespace bftile;
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

  __m128i * breordcolm = reinterpret_cast<__m128i*>(aligned_alloc(16*4*sizeof(int8_t), 1024)); // For easier visualisation

  //Sanity check
  gemmRowMColM(reinterpret_cast<int8_t *>(amat), reinterpret_cast<int8_t *>(bmat), 4, 16, 4, reinterpret_cast<int *>(cslow));

  // FUll pipeline test
  prepareBtile(bmat, breord);

  // For visualisation of the reordered format
  toColMajor(reinterpret_cast<int8_t *>(breord), reinterpret_cast<int8_t *>(breordcolm), 4, 16);

  multiplyTile(amat, breord, cres);
  multiplyTileEff(amat, breord, cresEff);

  //Compare the memory
  bool wrong = false;
  if (std::memcmp(cslow, cres, 4*4*sizeof(int32_t))) {
    std::cerr << "mm128 fast and slow gemm implementations differ" << std::endl;
    wrong = true;
  }
  if (std::memcmp(cslow, cresEff, 4*4*sizeof(int32_t))) {
    std::cerr << "mm128 fastEfficient and slow gemm implementations differ" << std::endl;
    wrong = true;
  }

  if (wrong || toprint) {
    std::cerr << "mm128 example" << std::endl;
    printMat(reinterpret_cast<int8_t *>(amat), 4, 16, "A int8", 2);
    printMat(reinterpret_cast<int8_t *>(bmatcolm), 16, 4, "B int8 colM", 2);
    printMat(reinterpret_cast<int8_t *>(breordcolm), 16, 4, "B int8 colMreord", 2);
    printMat(reinterpret_cast<int *>(cslow), 4, 4, "A * BcolM SlowMult", 5);
    printMat(reinterpret_cast<int32_t *>(&cres[0]), 4, 4, "A * Breord efficient", 5);
  }
  return wrong;
}

bool mm256Example(bool toprint=false) {
  using namespace bftile;
  using namespace bftile::mm256;
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

  __m256i * breordcolm = reinterpret_cast<__m256i*>(aligned_alloc(32*8*sizeof(int8_t), 1024)); // For easier visualisation

  //Sanity check
  gemmRowMColM(reinterpret_cast<int8_t *>(amat), reinterpret_cast<int8_t *>(bmat), 8, 32, 8, reinterpret_cast<int *>(cslow));

  // FUll pipeline test
  prepareBtile(bmat, breord);

  // For visualisation of the reordered format
  toColMajor(reinterpret_cast<int8_t *>(breord), reinterpret_cast<int8_t *>(breordcolm), 8, 32);

  multiplyTile(amat, breord, cres);

  //Compare the memory
  bool wrong = false;
  if (std::memcmp(cslow, cres, 8*8*sizeof(int32_t))) {
    std::cerr << "mm256 fast and slow gemm implementations differ" << std::endl;
    wrong = true;
  }
  if (wrong || toprint) {
    std::cerr << "mm256 example" << std::endl;
    printMat(reinterpret_cast<int8_t *>(amat), 8, 32, "A int8", 3);
    printMat(reinterpret_cast<int8_t *>(bmatcolm), 32, 8, "B int8 colM", 4);
    printMat(reinterpret_cast<int8_t *>(breordcolm), 32, 8, "B int8 colMreord", 4);
    printMat(reinterpret_cast<int *>(cslow), 8, 8, "A * BcolM SlowMult", 7);
    printMat(reinterpret_cast<int32_t *>(&cres[0]), 8, 8, "A * Breord efficient", 7);
  }
  return wrong;
}

bool mm512Example(bool toprint=false) {
  using namespace bftile;
  using namespace bftile::mm512;
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

  __m512i * breordcolm = reinterpret_cast<__m512i*>(aligned_alloc(64*16*sizeof(int8_t), 1024)); // For easier visualisation

  //Sanity check
  gemmRowMColM(reinterpret_cast<int8_t *>(amat), reinterpret_cast<int8_t *>(bmat), 16, 64, 16, reinterpret_cast<int *>(cslow));

  // FUll pipeline test
  prepareBtile(bmat, breord);

  // For visualisation of the reordered format
  toColMajor(reinterpret_cast<int8_t *>(breord), reinterpret_cast<int8_t *>(breordcolm), 16, 64);

  multiplyTile(amat, breord, cres);

  //Compare the memory
  bool wrong = false;
  if (std::memcmp(cslow, cres, 16*16*sizeof(int32_t))) {
    std::cerr << "mm512 fast and slow gemm implementations differ" << std::endl;
    wrong = true;
  }
  if (wrong || toprint) {
    std::cerr << "mm512 example" << std::endl;
    printMat(reinterpret_cast<int8_t *>(amat), 16, 64, "A int8", 3);
    printMat(reinterpret_cast<int8_t *>(bmatcolm), 64, 16, "B int8 colM", 4);
    printMat(reinterpret_cast<int8_t *>(breordcolm), 64, 16, "B int8 colMreord", 4);
    printMat(reinterpret_cast<int *>(cslow), 16, 16, "A * BcolM SlowMult", 7);
    printMat(reinterpret_cast<int32_t *>(&cres[0]), 16, 16, "A * Breord efficient", 7);
  }
  return wrong;
}

template<class gemmNS>
double GEMMTest(bftile::matrix dims, bool toprint=false) {
  using namespace bftile;
  size_t aRows = dims.aRows;
  size_t width = dims.width;
  size_t bCols = dims.bCols;

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

  gemmNS::prepareB::prepareBMatrix(B.begin(), BReord.begin(), width, bCols);

  //Sanity check
  gemmRowMColM(A.begin(), B.begin(), aRows, width, bCols, Cslow.begin());

  // For visualisation of the reordered format
  toColMajor(BReord.begin(), BReordColM.begin(), bCols, width);

  auto start = std::chrono::steady_clock::now();
  gemmNS::gemm::gemm(A.begin(), BReord.begin(), Cfast.begin(), aRows, width, bCols);
  auto end = std::chrono::steady_clock::now();

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
  std::chrono::duration<double> elapsed_seconds = end-start;
  return elapsed_seconds.count();
}

template<class gemmNS>
double gemmBenchmark(bftile::matrix dims) {
  using namespace bftile;
  size_t aRows = dims.aRows;
  size_t width = dims.width;
  size_t bCols = dims.bCols;

  AlignedVector<uint8_t> A(aRows*width);
  AlignedVector<int8_t> B(width*bCols);
  AlignedVector<int8_t> BReord(width*bCols);
  AlignedVector<int32_t> Cfast(aRows*bCols);

  for (size_t i = 0; i < aRows*width; i++) {
    A[i] = i % 255;
  }
  for (size_t i = 0; i < width*bCols; i++) {
    B[i] = i % 255;
  }
  for (auto&& item : Cfast) {
    item = 0;
  }

  // Prepare datapoints
  gemmNS::prepareB::prepareBMatrix(B.begin(), BReord.begin(), width, bCols);

  auto start = std::chrono::steady_clock::now();
  gemmNS::gemm::gemm(A.begin(), BReord.begin(), Cfast.begin(), aRows, width, bCols);
  auto end = std::chrono::steady_clock::now();

  doNotOptimizeAway(Cfast.begin());
  std::chrono::duration<double> elapsed_seconds = end-start;
  return elapsed_seconds.count();
}

void benchmark(size_t times=100) {
  double time_rows = 0;
  double time_width = 0;
  double time_width_addr = 0;
  double time_width_addr_loop = 0;
  double time_width_addr_loop_tile_loop = 0;
  double time_width_addr_loop_tile_loop_write_dep = 0;
  double time_width_addr_loop_tile_loop_write_dep_mm256 = 0;
  double time_width_addr_loop_tile_loop_write_dep_mm512 = 0;
  bftile::matrix matrices[11] = {{16, 64, 16},
                                 {16, 256, 256},
                                 {16, 2048, 256},
                                 {320, 256, 256},
                                 {480, 256, 256},
                                 {240, 256, 256},
                                 {208, 256, 256},
                                 {256, 256, 256},
                                 {1024, 1024, 1024},
                                 {4096, 4096, 128},
                                 {640, 320, 320}};
  for (size_t i = 0; i<times; i++) {
    for (auto&& matrix : matrices) {
      time_rows += gemmBenchmark<bftile::breadthfirst::runner>(matrix);
      time_width += gemmBenchmark<bftile::depthfirst::runner>(matrix);
      time_width_addr += gemmBenchmark<bftile::depthfirstaddr::runner>(matrix);
      time_width_addr_loop += gemmBenchmark<bftile::depthfirstaddrloop::runner>(matrix);
      time_width_addr_loop_tile_loop += gemmBenchmark<bftile::depthfirstaddrlooptileloop::runner>(matrix);
      time_width_addr_loop_tile_loop_write_dep += gemmBenchmark<bftile::depthfirstaddrlooptileloopwritedepend::runner>(matrix);
      time_width_addr_loop_tile_loop_write_dep_mm256 += gemmBenchmark<bftile::mm256::depthfirstaddrlooptileloopwritedepend::runner>(matrix);
      time_width_addr_loop_tile_loop_write_dep_mm512 += gemmBenchmark<bftile::mm512::depthfirstaddrlooptileloopwritedepend::runner>(matrix);
    }
  }
  std::cerr << "mm128 Iteration over rows-of-a took: " << time_rows << " seconds." << std::endl;
  std::cerr << "mm128 Iteration over width took: " << time_width << " seconds." << std::endl;
  std::cerr << "mm128 Iteration over width with addresses took: " << time_width_addr << " seconds." << std::endl;
  std::cerr << "mm128 Iteration over width with addresses assigned via for loop took: " << time_width_addr_loop << " seconds." << std::endl;
  std::cerr << "mm128 Iteration over width with addresses assigned via for loop, for loop tile took: " << time_width_addr_loop_tile_loop << " seconds." << std::endl;
  std::cerr << "mm128 Iteration over width with addresses assigned via for loop, for loop tile with write dependencies took: " << time_width_addr_loop_tile_loop_write_dep << " seconds." << std::endl;
  std::cerr << "mm256 Iteration over width with addresses assigned via for loop, for loop tile with write dependencies took: " << time_width_addr_loop_tile_loop_write_dep_mm256 << " seconds." << std::endl;
  std::cerr << "mm256 Iteration over width with addresses assigned via for loop, for loop tile with write dependencies took: " << time_width_addr_loop_tile_loop_write_dep_mm512 << " seconds." << std::endl;
}

int main() {
  mm128Example();
  mm256Example();
  mm512Example();
  bftile::matrix matricesmm128[14] = {{4, 16, 4},
                                 {4, 32, 4},
                                 {8, 32, 4},
                                 {64, 32, 8},
                                 {4, 16, 8},
                                 {4, 16, 32},
                                 {4, 16, 12},
                                 {640, 32, 32},
                                 {4, 48, 4},
                                 {640, 320, 32},
                                 {640, 320, 320},
                                 {256, 256, 4},
                                 {72, 320, 144},
                                 {4, 256, 512}};
  for (auto&& matrix : matricesmm128) {
    GEMMTest<bftile::breadthfirst::runner>(matrix);
    GEMMTest<bftile::depthfirst::runner>(matrix);
    GEMMTest<bftile::depthfirstaddr::runner>(matrix);
    GEMMTest<bftile::depthfirstaddrloop::runner>(matrix);
    GEMMTest<bftile::depthfirstaddrlooptileloop::runner>(matrix);
    GEMMTest<bftile::depthfirstaddrlooptileloopwritedepend::runner>(matrix);
  }

  bftile::matrix matricesmm256[14] = {{8, 32, 8},
                                 {8, 64, 8},
                                 {16, 32, 8},
                                 {8, 32, 16},
                                 {16, 64, 16},
                                 {16, 64, 64},
                                 {8, 32, 40},
                                 {640, 32, 32},
                                 {16, 96, 48},
                                 {640, 320, 32},
                                 {640, 320, 320},
                                 {256, 256, 8},
                                 {72, 320, 144},
                                 {40, 512, 512}};
  for (auto&& matrix : matricesmm256) {
    GEMMTest<bftile::mm256::depthfirstaddrlooptileloopwritedepend::runner>(matrix);
  }

  bftile::matrix matricesmm512[14] = {{16, 64, 16},
                                 {16, 128, 16},
                                 {32, 64, 16},
                                 {16, 64, 32},
                                 {32, 128, 32},
                                 {32, 128, 128},
                                 {16, 64, 80},
                                 {640, 64, 64},
                                 {16, 192, 48},
                                 {640, 320, 32},
                                 {640, 320, 320},
                                 {256, 256, 96},
                                 {64, 320, 144},
                                 {48, 512, 512}};
  for (auto&& matrix : matricesmm512) {
    GEMMTest<bftile::mm512::depthfirstaddrlooptileloopwritedepend::runner>(matrix);
  }
  benchmark(10);
}
