/*
 * SHALE embedded benchmark: lossless coding + ECLQ padding + Ascon-AEAD128, versus Ascon on raw PCM.
 * Build:  gcc -O3 -ffp-contract=off bench_main.c codec.c ascon.c -lm -o shale_bench
 * Run:    ./shale_bench input.wav            (16-bit PCM WAV, mono or stereo)
 * Prints MB/s, real-time factor and ciphertext size; checks bit-exact decryption.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

int shale_encode_frames(const int64_t *x, int64_t N, int C, int bits, int F, int maxp, int prec, int maxpo,
                        uint8_t *out, int64_t maxbytes, int64_t *lens);
int shale_decode_frame(const uint8_t *buf, int64_t len, int n, int C, int bits, int64_t *out);
int ascon_aead128_encrypt(uint8_t *c, const uint8_t *m, size_t mlen, const uint8_t *ad, size_t adlen, const uint8_t *npub, const uint8_t *k);
int ascon_aead128_decrypt(uint8_t *m, const uint8_t *c, size_t clen, const uint8_t *ad, size_t adlen, const uint8_t *npub, const uint8_t *k);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static int16_t *read_wav(const char *fn, int64_t *N, int *C, int *sr) {
    FILE *f = fopen(fn, "rb"); if (!f) return NULL;
    uint8_t h[12]; if (fread(h, 1, 12, f) != 12 || memcmp(h, "RIFF", 4) || memcmp(h + 8, "WAVE", 4)) return NULL;
    int bits = 0; int16_t *d = NULL;
    for (;;) {
        uint8_t ck[8]; if (fread(ck, 1, 8, f) != 8) break;
        uint32_t sz = ck[4] | ck[5] << 8 | ck[6] << 16 | (uint32_t)ck[7] << 24;
        if (!memcmp(ck, "fmt ", 4)) { uint8_t b[64]; size_t r = fread(b, 1, sz < 64 ? sz : 64, f); if (sz > 64) fseek(f, sz - 64, SEEK_CUR); (void)r;
            *C = b[2] | b[3] << 8; *sr = b[4] | b[5] << 8 | b[6] << 16 | b[7] << 24; bits = b[14] | b[15] << 8; }
        else if (!memcmp(ck, "data", 4)) { d = malloc(sz); if (fread(d, 1, sz, f) != sz) {} *N = sz / (2 * *C); break; }
        else fseek(f, sz + (sz & 1), SEEK_CUR);
    }
    fclose(f);
    return bits == 16 ? d : NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s file.wav [frame=4096] [reps=5]\n", argv[0]); return 1; }
    int64_t N; int C, sr; int16_t *pcm = read_wav(argv[1], &N, &C, &sr);
    if (!pcm) { fprintf(stderr, "need a 16-bit PCM WAV\n"); return 1; }
    int F = argc > 2 ? atoi(argv[2]) : 4096, reps = argc > 3 ? atoi(argv[3]) : 5;
    int64_t raw = N * C * 2; double secs = (double)N / sr;
    int64_t *x = malloc(N * C * sizeof(int64_t)); for (int64_t i = 0; i < N * C; i++) x[i] = pcm[i];
    int64_t nf = (N + F - 1) / F, mb = ((int64_t)F * C * 17 + 64 * C + 16) / 8 + 8;
    uint8_t *fr = malloc(nf * mb); int64_t *lens = malloc(nf * sizeof(int64_t));
    uint8_t key[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, nonce[16] = {0}, ad[25] = {0};
    uint8_t *pt = malloc(raw + 8 + 4 * nf + 64), *ct = malloc(raw + 8 + 4 * nf + 64 + 16), *dec = malloc(raw + 8 + 4 * nf + 64);
    double best_enc = 1e9, best_dec = 1e9, best_pcm = 1e9; size_t ptlen = 0;
    for (int r = 0; r < reps; r++) {
        double t0 = now();
        shale_encode_frames(x, N, C, 16, F, 12, 12, 8, fr, mb, lens);
        size_t p = 8 + 4 * nf;                         /* T | length table | payloads */
        for (int64_t i = 0; i < nf; i++) { memcpy(pt + 8 + 4 * i, &lens[i], 4); memcpy(pt + p, fr + i * mb, lens[i]); p += lens[i]; }
        memcpy(pt, &p, 8);
        size_t padded = p + (p / 8);                   /* example bucket: +12.5 % headroom, zero padding */
        if (padded > (size_t)(raw + 8 + 4 * nf)) padded = raw + 8 + 4 * nf;
        memset(pt + p, 0, padded - p); ptlen = padded;
        ascon_aead128_encrypt(ct, pt, ptlen, ad, sizeof ad, nonce, key);
        double t1 = now(); if (t1 - t0 < best_enc) best_enc = t1 - t0;
        /* decrypt + decode */
        t0 = now();
        if (ascon_aead128_decrypt(dec, ct, ptlen + 16, ad, sizeof ad, nonce, key)) { fprintf(stderr, "auth fail\n"); return 2; }
        int64_t *y = malloc((int64_t)F * C * sizeof(int64_t)); size_t q = 8 + 4 * nf; int ok = 1;
        for (int64_t i = 0; i < nf; i++) {
            int32_t L; memcpy(&L, dec + 8 + 4 * i, 4); int n = (int)((N - i * F) < F ? (N - i * F) : F);
            shale_decode_frame(dec + q, L, n, C, 16, y); q += L;
            for (int64_t k = 0; k < (int64_t)n * C; k++) if (y[k] != x[i * F * C + k]) ok = 0;
        }
        free(y);
        t1 = now(); if (t1 - t0 < best_dec) best_dec = t1 - t0;
        if (!ok) { fprintf(stderr, "NOT bit-exact\n"); return 3; }
        /* baseline: Ascon on raw PCM */
        t0 = now(); ascon_aead128_encrypt(ct, (uint8_t *)pcm, raw, ad, sizeof ad, nonce, key); t1 = now();
        if (t1 - t0 < best_pcm) best_pcm = t1 - t0;
    }
    printf("file %s: %.2f s, %d Hz, %d ch, %.2f MB\n", argv[1], secs, sr, C, raw / 1e6);
    printf("bit-exact decryption: yes\n");
    printf("SHALE encrypt  : %8.2f MB/s  (%.0fx real time)  ciphertext/PCM = %.3f\n", raw / best_enc / 1e6, secs / best_enc, (ptlen + 16.0) / raw);
    printf("SHALE decrypt  : %8.2f MB/s  (%.0fx real time)\n", raw / best_dec / 1e6, secs / best_dec);
    printf("Ascon raw PCM  : %8.2f MB/s  ciphertext/PCM = 1.000\n", raw / best_pcm / 1e6);
    return 0;
}
