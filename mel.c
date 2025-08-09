// log-mel spectrogram (128 x N frames)
// Written by Kenta Iwasaki. All rights reserved.
// 2025-08-10

/* compile with:
zig cc mel.c -o libmel.dylib -shared -Ofast -march=native -ffast-math \
             -funroll-loops -fopenmp -lomp \
             -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include
*/

#include <math.h>
#include <omp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Helpers / hints
#if defined(__GNUC__) || defined(__clang__)
#define ALIGN32 __attribute__((aligned(32)))
#define RESTRICT __restrict__
#define ALWAYS_INLINE inline __attribute__((always_inline, hot, flatten))
#define ASSUME_ALIGNED(p, a) __builtin_assume_aligned((p), (a))
#define LIKELY(x) __builtin_expect(!!(x), 1)
#else
#define ALIGN32
#define RESTRICT
#define ALWAYS_INLINE inline
#define ASSUME_ALIGNED(p, a) (p)
#define LIKELY(x) (x)
#endif

// Compile-time constants
#define N_FFT 400
#define HOP 160
#define BINS ((N_FFT >> 1) + 1) /* 201 */
#define INV_LN10 0.434294482f
#define EPS 1.0e-10f
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// External Mel tables
#include "mel_mapping_tables.h" // mel_row_ptr / mel_col_idx / mel_weight

// FFT precomputed tables
static float hann_window[N_FFT] ALIGN32;
static uint8_t bitrev8[8] ALIGN32;
static float W5nk_re[5][5] ALIGN32, W5nk_im[5][5] ALIGN32;
static float W8_re[8] ALIGN32, W8_im[8] ALIGN32;
static float W25_re[5][5] ALIGN32, W25_im[5][5] ALIGN32;
static float WN200_re[25][8] ALIGN32, WN200_im[25][8] ALIGN32;
static float W400_re[BINS] ALIGN32, W400_im[BINS] ALIGN32;

// Thread-local scratch buffers
typedef struct {
  float windowed[N_FFT] ALIGN32;
  float y_re[N_FFT] ALIGN32; // 25 x 16 grid, row-major by n2 then k1
  float y_im[N_FFT] ALIGN32;
  float fft_re[N_FFT] ALIGN32; // full 400-point FFT result
  float fft_im[N_FFT] ALIGN32;
} Scratch;

static _Thread_local Scratch scratch_buffer ALIGN32;
static ALWAYS_INLINE Scratch *get_scratch(void) { return &scratch_buffer; }

static ALWAYS_INLINE float fast_log10f(float x) { return logf(x) * INV_LN10; }

#define BUTTER(i0, i1, WR, WI)                                                 \
  do {                                                                         \
    float ur = xr[(i0)];                                                       \
    float ui = xi[(i0)];                                                       \
    float vr = xr[(i1)];                                                       \
    float vi = xi[(i1)];                                                       \
    float trr = vr * (WR) - vi * (WI);                                         \
    float tri = vr * (WI) + vi * (WR);                                         \
    xr[(i0)] = ur + trr;                                                       \
    xi[(i0)] = ui + tri;                                                       \
    xr[(i1)] = ur - trr;                                                       \
    xi[(i1)] = ui - tri;                                                       \
  } while (0)

// 8-point in-place complex FFT (DIT, bit-reversed input)
static ALWAYS_INLINE void fft8(float *RESTRICT xr, float *RESTRICT xi) {
  // Bit-reverse permutation (swap in-place where i < br(i))
#pragma omp simd aligned(xr, xi : 32)
  for (int i = 0; i < 8; ++i) {
    int j = bitrev8[i];
    if (j > i) {
      float tr = xr[i];
      float ti = xi[i];
      xr[i] = xr[j];
      xi[i] = xi[j];
      xr[j] = tr;
      xi[j] = ti;
    }
  }
// m=2
#pragma omp simd aligned(xr, xi : 32)
  for (int k = 0; k < 8; k += 2) {
    BUTTER(k + 0, k + 1, 1.0f, 0.0f);
  }
// m=4 (idx {0,2})
#pragma omp simd aligned(xr, xi : 32)
  for (int k = 0; k < 8; k += 4) {
    BUTTER(k + 0, k + 2, W8_re[0], W8_im[0]);
    BUTTER(k + 1, k + 3, W8_re[2], W8_im[2]);
  }
  // m=8 (idx {0..3})
  {
    int k = 0;
    BUTTER(k + 0, k + 4, W8_re[0], W8_im[0]);
    BUTTER(k + 1, k + 5, W8_re[1], W8_im[1]);
    BUTTER(k + 2, k + 6, W8_re[2], W8_im[2]);
    BUTTER(k + 3, k + 7, W8_re[3], W8_im[3]);
  }
}

// 25-point FFT via 5x5 Cooley-Tukey
static ALWAYS_INLINE void
fft25_column_full_into(const float *RESTRICT y_r, const float *RESTRICT y_i,
                       int stride, int k1, int out_k1_period,
                       float *RESTRICT fft_r, float *RESTRICT fft_i) {
  // For each s in 0..4, compute DFT-5 over r (n2 = 5*r + s)
  float A_r[25] ALIGN32;
  float A_i[25] ALIGN32;
  for (int s = 0; s < 5; ++s) {
    float in_r[5] ALIGN32, in_i[5] ALIGN32;
#pragma omp simd aligned(in_r, in_i, y_r, y_i : 32)
    for (int r = 0; r < 5; ++r) {
      int n2 = r * 5 + s;
      int idx = n2 * stride;
      in_r[r] = y_r[idx];
      in_i[r] = y_i[idx];
    }
    // DFT-5 over r -> k0 (inner loop unrolled over r)
#pragma omp simd aligned(in_r, in_i, A_r, A_i, W5nk_re, W5nk_im : 32)
    for (int k0 = 0; k0 < 5; ++k0) {
      float sr = 0.f, si = 0.f;
      sr += in_r[0] * W5nk_re[0][k0] - in_i[0] * W5nk_im[0][k0];
      si += in_r[0] * W5nk_im[0][k0] + in_i[0] * W5nk_re[0][k0];
      sr += in_r[1] * W5nk_re[1][k0] - in_i[1] * W5nk_im[1][k0];
      si += in_r[1] * W5nk_im[1][k0] + in_i[1] * W5nk_re[1][k0];
      sr += in_r[2] * W5nk_re[2][k0] - in_i[2] * W5nk_im[2][k0];
      si += in_r[2] * W5nk_im[2][k0] + in_i[2] * W5nk_re[2][k0];
      sr += in_r[3] * W5nk_re[3][k0] - in_i[3] * W5nk_im[3][k0];
      si += in_r[3] * W5nk_im[3][k0] + in_i[3] * W5nk_re[3][k0];
      sr += in_r[4] * W5nk_re[4][k0] - in_i[4] * W5nk_im[4][k0];
      si += in_r[4] * W5nk_im[4][k0] + in_i[4] * W5nk_re[4][k0];
      float twr = W25_re[s][k0];
      float twi = W25_im[s][k0];
      int idxA = s * 5 + k0;
      A_r[idxA] = sr * twr - si * twi;
      A_i[idxA] = sr * twi + si * twr;
    }
  }

  // For each k0 in 0..4, DFT-5 over s to produce k2 = k0 + 5*k2b, k2b 0..4
  for (int k0 = 0; k0 < 5; ++k0) {
    // Inputs A[s*5 + k0] over s
#pragma omp simd aligned(A_r, A_i, fft_r, fft_i, W5nk_re, W5nk_im : 32)
    for (int k2b = 0; k2b < 5; ++k2b) {
      float sr = 0.f, si = 0.f;
      int i0 = 0 * 5 + k0;
      int i1 = 1 * 5 + k0;
      int i2 = 2 * 5 + k0;
      int i3 = 3 * 5 + k0;
      int i4 = 4 * 5 + k0;
      sr += A_r[i0] * W5nk_re[0][k2b] - A_i[i0] * W5nk_im[0][k2b];
      si += A_r[i0] * W5nk_im[0][k2b] + A_i[i0] * W5nk_re[0][k2b];
      sr += A_r[i1] * W5nk_re[1][k2b] - A_i[i1] * W5nk_im[1][k2b];
      si += A_r[i1] * W5nk_im[1][k2b] + A_i[i1] * W5nk_re[1][k2b];
      sr += A_r[i2] * W5nk_re[2][k2b] - A_i[i2] * W5nk_im[2][k2b];
      si += A_r[i2] * W5nk_im[2][k2b] + A_i[i2] * W5nk_re[2][k2b];
      sr += A_r[i3] * W5nk_re[3][k2b] - A_i[i3] * W5nk_im[3][k2b];
      si += A_r[i3] * W5nk_im[3][k2b] + A_i[i3] * W5nk_re[3][k2b];
      sr += A_r[i4] * W5nk_re[4][k2b] - A_i[i4] * W5nk_im[4][k2b];
      si += A_r[i4] * W5nk_im[4][k2b] + A_i[i4] * W5nk_re[4][k2b];
      int k2 = k0 + 5 * k2b; // 0..24
      int K = k1 + out_k1_period * k2;
      fft_r[K] = sr;
      fft_i[K] = si;
    }
  }
}

// Complex FFT of length 200 for real input (imag assumed 0), using 8x25
// factorization.
static ALWAYS_INLINE void cfft200_real_input(const float *RESTRICT x,
                                             float *RESTRICT out_re,
                                             float *RESTRICT out_im,
                                             Scratch *RESTRICT s) {
  float *RESTRICT y_r = s->y_re; // reuse scratch (size >= 200)
  float *RESTRICT y_i = s->y_im;
  // For each n2=0..24, 8-pt FFT on x[n2 + 25*n1 * stride]
  for (int n2 = 0; n2 < 25; ++n2) {
    float xr[8] ALIGN32, xi[8] ALIGN32;
#pragma omp simd aligned(xr, xi, x : 32)
    for (int n1 = 0; n1 < 8; ++n1) {
      int idx = (n2 + 25 * n1) * 2;
      xr[n1] = x[idx];
      xi[n1] = 0.0f;
    }
    fft8(xr, xi);
    float *row_r = y_r + n2 * 8;
    float *row_i = y_i + n2 * 8;
#pragma omp simd aligned(row_r, row_i, xr, xi, WN200_re, WN200_im : 32)
    for (int k1 = 0; k1 < 8; ++k1) {
      float wr = WN200_re[n2][k1];
      float wi = WN200_im[n2][k1];
      float ar = xr[k1];
      float ai = xi[k1];
      row_r[k1] = ar * wr - ai * wi;
      row_i[k1] = ar * wi + ai * wr;
    }
  }
  // For each k1, 25-pt FFT down column, output to out_re/out_im (len 200)
  for (int k1 = 0; k1 < 8; ++k1) {
    const float *col_r = y_r + k1; // stride 8 over n2
    const float *col_i = y_i + k1;
    fft25_column_full_into(col_r, col_i, 8, k1, 8, out_re, out_im);
  }
}

// Power spectrum using real FFT (RFFT) for N=400
static ALWAYS_INLINE void power_spectrum_400(const float *RESTRICT x,
                                             float *RESTRICT power_out,
                                             Scratch *RESTRICT s) {
  // Compute 200-pt complex FFTs by reading x with stride 2 (evens and odds)
  float *RESTRICT fr = s->fft_re;
  float *RESTRICT fi = s->fft_im;
  // even indices: start=0, stride=2
  cfft200_real_input(x, fr + 0, fi + 0, s);
  // odd indices: start=1 (we pass x+1 pointer), stride=2
  cfft200_real_input(x + 1, fr + 200, fi + 200, s);

  // Combine to get X[k] for k=0..200 and compute power
  // k=0
  {
    const float xer = fr[0], xei = fi[0];
    const float xor_ = fr[200], xoi = fi[200];
    const float Re = xer + xor_;
    const float Im = xei + xoi;
    power_out[0] = Re * Re + Im * Im;
  }
// k=1..199
#pragma omp simd aligned(fr, fi, power_out, W400_re, W400_im : 32)
  for (int k = 1; k < 200; ++k) {
    const float xer = fr[k], xei = fi[k];
    const float xor_ = fr[200 + k], xoi = fi[200 + k];
    const float wr = W400_re[k];
    const float wi = W400_im[k];
    const float tr = xor_ * wr - xoi * wi;
    const float ti = xor_ * wi + xoi * wr;
    const float Re = xer + tr;
    const float Im = xei + ti;
    power_out[k] = Re * Re + Im * Im;
  }
  // k=200 (Nyquist)
  {
    const float xer0 = fr[0], xei0 = fi[0];
    const float xor0 = fr[200], xoi0 = fi[200];
    const float Re = xer0 - xor0; // W^200 = -1
    const float Im = xei0 - xoi0;
    power_out[200] = Re * Re + Im * Im;
  }
}

static ALWAYS_INLINE float process_frame(const float *RESTRICT audio,
                                         int audio_len, int base,
                                         float *RESTRICT out_flat,
                                         int frame_idx, int n_frames) {
  Scratch *s = get_scratch();
  float *RESTRICT windowed = (float *)ASSUME_ALIGNED(s->windowed, 32);

  const int start = base;
  const int end = base + N_FFT;

  // Apply Hann window with zero-padding for out-of-bound samples

  if (LIKELY(start >= 0 && end <= audio_len)) {
    const float *src = audio + start;
#pragma omp simd aligned(windowed, hann_window : 32)
    for (int k = 0; k < N_FFT; ++k) {
      windowed[k] = src[k] * hann_window[k];
    }
  } else {
#pragma omp simd aligned(windowed, hann_window : 32)
    for (int k = 0; k < N_FFT; ++k) {
      int idx = base + k;
      float sample = (idx >= 0 && idx < audio_len) ? audio[idx] : 0.f;
      windowed[k] = sample * hann_window[k];
    }
  }

  // Compute power spectrum via real FFT
  power_spectrum_400(windowed, windowed, s);

  // Mel dot-product
  const int stride = n_frames;
  float *dst = out_flat + frame_idx;
  float g_max = -1e30f;

  for (int m = 0; m < 128; ++m, dst += stride) {
    float sum = 0.f;
    int p = mel_row_ptr[m];
    int end_p = mel_row_ptr[m + 1];

#pragma omp simd aligned(windowed : 32) reduction(+ : sum)
    for (int idx = p; idx < end_p; ++idx)
      sum += windowed[mel_col_idx[idx]] * mel_weight[idx];

    const float v = fast_log10f(sum < EPS ? EPS : sum);
    *dst = v;
    if (v > g_max)
      g_max = v;
  }
  return g_max;
}

void log_mel_spectrogram(const float *RESTRICT audio, int audio_len,
                         int pad_tail, float *RESTRICT out_flat) {
  const int pad = N_FFT >> 1;
  const int n_frames = (audio_len + pad_tail) / HOP;
  const int N = 128 * n_frames;

  float g_max = -1e30f;
#pragma omp parallel for schedule(static, 64) reduction(max : g_max)
  for (int t = 0; t < n_frames; ++t) {
    int base = t * HOP - pad;
    float local = process_frame(audio, audio_len, base, out_flat, t, n_frames);
    if (local > g_max)
      g_max = local;
  }

  // Clamp & normalise in a single pass
  const float thr = g_max - 8.f;
#pragma omp parallel for simd schedule(static, 512)
  for (int i = 0; i < N; ++i) {
    float v = out_flat[i];
    v = (v < thr ? thr : v);         /* clamp */
    out_flat[i] = (v + 4.f) * 0.25f; /* normalise */
  }
}

void init_globals(void) {
  // Periodic Hann window
  for (int i = 0; i < N_FFT; ++i) {
    hann_window[i] = 0.5f * (1.f - cosf((2.f * M_PI * i) / N_FFT));
  }

  // Bit-reverse table for 8 (3-bit reverse)
  for (int i = 0; i < 8; ++i) {
    int b0 = (i >> 0) & 1;
    int b1 = (i >> 1) & 1;
    int b2 = (i >> 2) & 1;
    bitrev8[i] = (uint8_t)((b0 << 2) | (b1 << 1) | (b2 << 0));
  }

  // W8 roots: e^{-2π i k / 8}
  for (int k = 0; k < 8; ++k) {
    float angle = -2.0f * M_PI * k / 8.0f;
    W8_re[k] = cosf(angle);
    W8_im[k] = sinf(angle);
  }

  // WN200 twiddles: e^{-2π i n2*k1 / 200} for n2 in 0..24, k1 in 0..7
  for (int n2 = 0; n2 < 25; ++n2) {
    for (int k1 = 0; k1 < 8; ++k1) {
      float angle = -2.0f * M_PI * (float)(n2 * k1) / 200.0f;
      WN200_re[n2][k1] = cosf(angle);
      WN200_im[n2][k1] = sinf(angle);
    }
  }

  // W400 twiddles for RFFT combine: W400[k] = e^{-2π i k / 400}, k=0..200
  for (int k = 0; k <= 200; ++k) {
    float angle = -2.0f * M_PI * (float)k / 400.0f;
    W400_re[k] = cosf(angle);
    W400_im[k] = sinf(angle);
  }

  // W5 and W25 twiddles for 25-point stage
  for (int n = 0; n < 5; ++n) {
    for (int k = 0; k < 5; ++k) {
      float angle5 = -2.0f * M_PI * (float)(n * k) / 5.0f;
      W5nk_re[n][k] = cosf(angle5);
      W5nk_im[n][k] = sinf(angle5);
    }
  }
  for (int r = 0; r < 5; ++r) {
    for (int k = 0; k < 5; ++k) {
      float angle25 = -2.0f * M_PI * (float)(r * k) / 25.0f;
      W25_re[r][k] = cosf(angle25);
      W25_im[r][k] = sinf(angle25);
    }
  }
}