/*
 * SHALE on ESP8266 (NodeMCU) - port of the ESP32 benchmark.
 * Lossless coder + ECLQ padding + Ascon-AEAD128, stream mode, 20 ms records,
 * on a real 16 kHz speech clip (CREMA-D) stored in flash. Checks bit-exactness.
 * Only the 20 ms mode is run: the 100 ms mode needs more RAM than an ESP8266 has.
 * Serial Monitor: 115200 baud.
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
static const int B320[] = {492, 644};   /* ECLQ buckets, 20 ms records */
static const int F = 320;

static int bucket(int L) { for (int i = 0; i < 2; i++) if (L <= B320[i]) return B320[i]; return B320[1]; }
static inline int16_t clip_at(int i) { return (int16_t)pgm_read_word(&CLIP[i]); }

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("\n===== SHALE ESP8266 benchmark =====");
  Serial.printf("chip id %08x, CPU %d MHz, free heap %u, largest block %u\n", ESP.getChipId(), ESP.getCpuFreqMHz(), ESP.getFreeHeap(), ESP.getMaxFreeBlockSize());
  Serial.printf("clip: %d samples, %.2f s, 16 kHz mono (CREMA-D). Running, please wait (1-3 min)...\n", CLIP_N, CLIP_N / 16000.0);

  const int maxb = (F * 17 + 64 + 16) / 8 + 8, top = 2 * F + 4;
  int64_t *x = (int64_t *)malloc(F * sizeof(int64_t));
  int64_t *y = (int64_t *)malloc(F * sizeof(int64_t));
  uint8_t *fr = (uint8_t *)malloc(maxb), *pt = (uint8_t *)malloc(top), *ct = (uint8_t *)malloc(top + 16), *rx = (uint8_t *)malloc(top);
  int16_t *raw = (int16_t *)malloc(F * 2);
  if (!x || !y || !fr || !pt || !ct || !rx || !raw) { Serial.println("ERROR: out of memory"); return; }

  int64_t len; uint8_t nonce[16], ad[25] = {0};
  int frames = 0, auth_fail = 0, bad = 0; bool exact = true;
  double enc_us = 0, dec_us = 0, pcm_us = 0; long bytes_sent = 0, bytes_pcm = 0;
  uint32_t minheap = ESP.getFreeHeap();

  for (int s = 0, i = 0; s < CLIP_N; s += F, i++) {
    int n = min(F, CLIP_N - s);
    for (int k = 0; k < n; k++) x[k] = clip_at(s + k);
    memset(nonce, 0, 16); memcpy(nonce, &i, 4); nonce[8] = (s + F >= CLIP_N);
    memcpy(ad + 16, &i, 4); ad[24] = nonce[8];
    uint32_t t0 = micros();
    shale_encode_frames(x, n, 1, 16, F, 12, 12, 8, fr, maxb, &len);
    uint32_t L = 4 + len, flag = 0;
    if (len >= 2 * n) { L = 4 + 2 * n; flag = 0x80000000u; }
    int P = bucket(L);
    uint32_t lf = (L - 4) | flag; memcpy(pt, &lf, 4);
    if (flag) for (int k = 0; k < n; k++) { int16_t v = (int16_t)x[k]; memcpy(pt + 4 + 2 * k, &v, 2); }
    else memcpy(pt + 4, fr, len);
    memset(pt + L, 0, P - L);
    ascon_aead128_encrypt(ct, pt, P, ad, sizeof ad, nonce, KEY);
    uint32_t t1 = micros();
    if (ascon_aead128_decrypt(rx, ct, P + 16, ad, sizeof ad, nonce, KEY)) auth_fail++;
    uint32_t lf2; memcpy(&lf2, rx, 4);
    if (lf2 & 0x80000000u) for (int k = 0; k < n; k++) { int16_t v; memcpy(&v, rx + 4 + 2 * k, 2); y[k] = v; }
    else shale_decode_frame(rx + 4, lf2, n, 1, 16, y);
    uint32_t t2 = micros();
    for (int k = 0; k < n; k++) if (y[k] != x[k]) { exact = false; bad++; }
    for (int k = 0; k < n; k++) raw[k] = clip_at(s + k);
    uint32_t t3 = micros();
    ascon_aead128_encrypt(ct, (uint8_t *)raw, 2 * n, ad, sizeof ad, nonce, KEY);
    uint32_t t4 = micros();
    enc_us += t1 - t0; dec_us += t2 - t1; pcm_us += t4 - t3;
    bytes_sent += P + 16; bytes_pcm += 2 * n + 16; frames++;
    uint32_t h = ESP.getFreeHeap(); if (h < minheap) minheap = h;
    yield();                                    /* keep the watchdog happy */
    if (frames % 20 == 0) Serial.printf("  ... %d frames\n", frames);
  }
  double audio_us = 1e6 * CLIP_N / 16000.0;
  Serial.println("--- stream mode, 320-sample records (20 ms) ---");
  Serial.printf("frames %d, bit-exact %s (%d bad samples), auth failures %d\n", frames, exact ? "YES" : "NO", bad, auth_fail);
  Serial.printf("SHALE encrypt per frame : %.3f ms  (CPU load %.1f %%, %.2fx real time)\n", enc_us / frames / 1000, 100 * enc_us / audio_us, audio_us / enc_us);
  Serial.printf("SHALE decrypt per frame : %.3f ms  (CPU load %.1f %%, %.2fx real time)\n", dec_us / frames / 1000, 100 * dec_us / audio_us, audio_us / dec_us);
  Serial.printf("Ascon raw PCM per frame : %.3f ms\n", pcm_us / frames / 1000);
  Serial.printf("bytes sent SHALE %ld vs raw PCM+tag %ld  (ratio %.3f)\n", bytes_sent, bytes_pcm, (double)bytes_sent / bytes_pcm);
  Serial.printf("min free heap during run %u bytes\n", minheap);
  Serial.println("===== done: please copy everything above =====");
}

void loop() { delay(1000); }
