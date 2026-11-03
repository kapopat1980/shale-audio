/*
 * SHALE on ESP32 (Arduino IDE): lossless coder + ECLQ padding + Ascon-AEAD128, stream mode,
 * on a real 16 kHz speech clip stored in flash. Prints timing, real-time factor, CPU load,
 * bytes on air and a bit-exact check. Open Serial Monitor at 115200 baud and copy the output.
 */
#include <Arduino.h>
#include <string.h>
#include "clip.h"

extern "C" {
int shale_encode_frames(const int64_t *x, int64_t N, int C, int bits, int F, int maxp, int prec, int maxpo,
                        uint8_t *out, int64_t maxbytes, int64_t *lens);
int shale_decode_frame(const uint8_t *buf, int64_t len, int n, int C, int bits, int64_t *out);
int ascon_aead128_encrypt(uint8_t *c, const uint8_t *m, size_t mlen, const uint8_t *ad, size_t adlen, const uint8_t *npub, const uint8_t *k);
int ascon_aead128_decrypt(uint8_t *m, const uint8_t *c, size_t clen, const uint8_t *ad, size_t adlen, const uint8_t *npub, const uint8_t *k);
}

static const uint8_t KEY[16] = {0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE, 1, 2, 3, 4, 5, 6, 7, 8};

struct Result { int F; int frames; double enc_us, dec_us, pcm_enc_us; long bytes_sent, bytes_pcm; bool exact; int auth_fail; };

/* ECLQ buckets (record sizes in bytes incl. 4-byte length field) designed on the public VCTK profile */
static const int B320[] = {492, 644};                 /* 20 ms, H <= 0.1 bit  */
static const int B1600[] = {2416, 2688, 2704, 3204};  /* 100 ms, H <= 0.1 bit */

static int bucket(int F, int L) {
  const int *b = F == 320 ? B320 : B1600; int n = F == 320 ? 2 : 4;
  for (int i = 0; i < n; i++) if (L <= b[i]) return b[i];
  return b[n - 1];
}

static Result run(int F) {
  Result R = {F, 0, 0, 0, 0, 0, 0, true, 0};
  const int maxb = (F * 17 + 64 + 16) / 8 + 8, top = 2 * F + 4;
  int64_t *x = (int64_t *)malloc(F * sizeof(int64_t));
  int64_t *y = (int64_t *)malloc(F * sizeof(int64_t));
  uint8_t *fr = (uint8_t *)malloc(maxb), *pt = (uint8_t *)malloc(top), *ct = (uint8_t *)malloc(top + 16), *rx = (uint8_t *)malloc(top);
  int64_t len;
  uint8_t nonce[16], ad[25] = {0};
  for (int s = 0, i = 0; s < CLIP_N; s += F, i++) {
    int n = min(F, CLIP_N - s);
    for (int k = 0; k < n; k++) x[k] = CLIP[s + k];
    memset(nonce, 0, 16); memcpy(nonce, &i, 4); nonce[8] = (s + F >= CLIP_N);
    memcpy(ad + 16, &i, 4); ad[24] = nonce[8];
    /* ---- sender: code, pad, encrypt ---- */
    uint32_t t0 = micros();
    shale_encode_frames(x, n, 1, 16, F, 12, 12, 8, fr, maxb, &len);
    uint32_t L = 4 + len, flag = 0;
    if (len >= 2 * n) { L = 4 + 2 * n; flag = 0x80000000u; }        /* verbatim fallback */
    int P = bucket(F, L);
    uint32_t lf = (L - 4) | flag; memcpy(pt, &lf, 4);
    if (flag) for (int k = 0; k < n; k++) { int16_t v = (int16_t)x[k]; memcpy(pt + 4 + 2 * k, &v, 2); }
    else memcpy(pt + 4, fr, len);
    memset(pt + L, 0, P - L);
    ascon_aead128_encrypt(ct, pt, P, ad, sizeof ad, nonce, KEY);
    uint32_t t1 = micros();
    /* ---- receiver: verify, decrypt, decode ---- */
    if (ascon_aead128_decrypt(rx, ct, P + 16, ad, sizeof ad, nonce, KEY)) R.auth_fail++;
    uint32_t lf2; memcpy(&lf2, rx, 4);
    if (lf2 & 0x80000000u) for (int k = 0; k < n; k++) { int16_t v; memcpy(&v, rx + 4 + 2 * k, 2); y[k] = v; }
    else shale_decode_frame(rx + 4, lf2, n, 1, 16, y);
    uint32_t t2 = micros();
    for (int k = 0; k < n; k++) if (y[k] != x[k]) R.exact = false;
    /* ---- baseline: Ascon on raw PCM frame ---- */
    int16_t raw[1600]; for (int k = 0; k < n; k++) raw[k] = CLIP[s + k];
    uint32_t t3 = micros();
    ascon_aead128_encrypt(ct, (uint8_t *)raw, 2 * n, ad, sizeof ad, nonce, KEY);
    uint32_t t4 = micros();
    R.enc_us += t1 - t0; R.dec_us += t2 - t1; R.pcm_enc_us += t4 - t3;
    R.bytes_sent += P + 16; R.bytes_pcm += 2 * n + 16; R.frames++;
  }
  free(x); free(y); free(fr); free(pt); free(ct); free(rx);
  return R;
}

static void report(const Result &R) {
  double audio_us = 1e6 * CLIP_N / 16000.0, frame_ms = 1000.0 * R.F / 16000.0;
  Serial.printf("--- stream mode, %d-sample records (%.0f ms) ---\n", R.F, frame_ms);
  Serial.printf("frames %d, bit-exact %s, auth failures %d\n", R.frames, R.exact ? "YES" : "NO", R.auth_fail);
  Serial.printf("SHALE encrypt per frame : %.3f ms  (CPU load %.1f %%, %.1fx real time)\n", R.enc_us / R.frames / 1000, 100 * R.enc_us / audio_us, audio_us / R.enc_us);
  Serial.printf("SHALE decrypt per frame : %.3f ms  (CPU load %.1f %%, %.1fx real time)\n", R.dec_us / R.frames / 1000, 100 * R.dec_us / audio_us, audio_us / R.dec_us);
  Serial.printf("Ascon raw PCM per frame : %.3f ms\n", R.pcm_enc_us / R.frames / 1000);
  Serial.printf("bytes sent SHALE %ld vs raw PCM+tag %ld  (ratio %.3f)\n", R.bytes_sent, R.bytes_pcm, (double)R.bytes_sent / R.bytes_pcm);
}

static void bench_task(void *) {
  delay(1500);
  Serial.println("\n===== SHALE ESP32 benchmark =====");
  Serial.printf("chip %s rev %d, %d core(s), CPU %d MHz, free heap %u bytes\n", ESP.getChipModel(), ESP.getChipRevision(), ESP.getChipCores(), ESP.getCpuFreqMHz(), ESP.getFreeHeap());
  Serial.printf("clip: %d samples, %.2f s, 16 kHz mono (CREMA-D)\n", CLIP_N, CLIP_N / 16000.0);
  /* Ascon throughput */
  static uint8_t buf[4096], out[4096 + 16]; uint8_t nonce[16] = {0}, ad[1] = {0};
  for (int i = 0; i < 4096; i++) buf[i] = i * 7;
  uint32_t t0 = micros();
  for (int r = 0; r < 20; r++) ascon_aead128_encrypt(out, buf, 4096, ad, 0, nonce, KEY);
  uint32_t dt = micros() - t0;
  Serial.printf("Ascon-AEAD128: %.3f MB/s, %.1f cycles/byte\n", 20 * 4096.0 / dt, (double)dt * ESP.getCpuFreqMHz() / (20 * 4096.0));
  for (int rep = 0; rep < 2; rep++) {           /* first pass warms caches; report the second */
    Result a = run(320), b = run(1600);
    if (rep == 1) { report(a); report(b); }
  }
  Serial.printf("free heap after run %u bytes, min free heap %u bytes\n", ESP.getFreeHeap(), ESP.getMinFreeHeap());
  Serial.println("===== done: please copy everything above =====");
  vTaskDelete(NULL);
}

void setup() {
  Serial.begin(115200);
  xTaskCreate(bench_task, "shale", 32768, NULL, 1, NULL);   /* 32 KB stack, any core (works on single-core ESP32-C3/S2 too) */
}

void loop() { delay(1000); }
