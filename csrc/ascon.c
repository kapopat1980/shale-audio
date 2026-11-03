/*
 * Ascon-AEAD128 and Ascon-XOF128 / Ascon-CXOF128 (NIST SP 800-232),
 * written for SHALE from the specification. Portable C99, 64-bit words,
 * little-endian byte order as required by SP 800-232.
 * Verified against the official LWC KAT files (see tests/test_ascon.py).
 */
#include <stdint.h>
#include <string.h>
#include <stddef.h>

#define ROR(x, n) (((x) >> (n)) | ((x) << (64 - (n))))

static inline uint64_t load64(const uint8_t *p, size_t n) {
    uint64_t x = 0;
    for (size_t i = 0; i < n; i++) x |= (uint64_t)p[i] << (8 * i);
    return x;
}
static inline void store64(uint8_t *p, uint64_t x, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t)(x >> (8 * i));
}
static inline uint64_t pad(size_t i) { return 0x01ULL << (8 * i); }
static inline uint64_t mask(size_t n) { return n >= 8 ? ~0ULL : ((1ULL << (8 * n)) - 1); }

typedef struct { uint64_t x[5]; } state_t;

static inline void round_(state_t *s, uint64_t c) {
    uint64_t x0 = s->x[0], x1 = s->x[1], x2 = s->x[2], x3 = s->x[3], x4 = s->x[4];
    uint64_t t0, t1, t2, t3, t4;
    x2 ^= c;
    x0 ^= x4; x4 ^= x3; x2 ^= x1;
    t0 = ~x0 & x1; t1 = ~x1 & x2; t2 = ~x2 & x3; t3 = ~x3 & x4; t4 = ~x4 & x0;
    x0 ^= t1; x1 ^= t2; x2 ^= t3; x3 ^= t4; x4 ^= t0;
    x1 ^= x0; x0 ^= x4; x3 ^= x2; x2 = ~x2;
    x0 ^= ROR(x0, 19) ^ ROR(x0, 28);
    x1 ^= ROR(x1, 61) ^ ROR(x1, 39);
    x2 ^= ROR(x2, 1) ^ ROR(x2, 6);
    x3 ^= ROR(x3, 10) ^ ROR(x3, 17);
    x4 ^= ROR(x4, 7) ^ ROR(x4, 41);
    s->x[0] = x0; s->x[1] = x1; s->x[2] = x2; s->x[3] = x3; s->x[4] = x4;
}

static inline void perm(state_t *s, int rounds) {
    for (int r = 12 - rounds; r < 12; r++)
        round_(s, (uint64_t)(0xf0 - r * 0x10 + r));
}

/* ---------------- Ascon-AEAD128 ---------------- */
#define AEAD_IV 0x00001000808c0001ULL

static void aead_init(state_t *s, const uint8_t *k, const uint8_t *n,
                      uint64_t *k0, uint64_t *k1) {
    *k0 = load64(k, 8); *k1 = load64(k + 8, 8);
    s->x[0] = AEAD_IV; s->x[1] = *k0; s->x[2] = *k1;
    s->x[3] = load64(n, 8); s->x[4] = load64(n + 8, 8);
    perm(s, 12);
    s->x[3] ^= *k0; s->x[4] ^= *k1;
}

static void aead_ad(state_t *s, const uint8_t *ad, size_t adlen) {
    if (adlen) {
        while (adlen >= 16) {
            s->x[0] ^= load64(ad, 8); s->x[1] ^= load64(ad + 8, 8);
            perm(s, 8); ad += 16; adlen -= 16;
        }
        if (adlen >= 8) {
            s->x[0] ^= load64(ad, 8);
            s->x[1] ^= load64(ad + 8, adlen - 8) ^ pad(adlen - 8);
        } else {
            s->x[0] ^= load64(ad, adlen) ^ pad(adlen);
        }
        perm(s, 8);
    }
    s->x[4] ^= 0x8000000000000000ULL;
}

static void aead_final(state_t *s, uint64_t k0, uint64_t k1, uint8_t *tag) {
    s->x[2] ^= k0; s->x[3] ^= k1;
    perm(s, 12);
    s->x[3] ^= k0; s->x[4] ^= k1;
    store64(tag, s->x[3], 8); store64(tag + 8, s->x[4], 8);
}

/* c must hold mlen + 16 bytes. returns 0 */
int ascon_aead128_encrypt(uint8_t *c, const uint8_t *m, size_t mlen,
                          const uint8_t *ad, size_t adlen,
                          const uint8_t *npub, const uint8_t *k) {
    state_t s; uint64_t k0, k1;
    aead_init(&s, k, npub, &k0, &k1);
    aead_ad(&s, ad, adlen);
    while (mlen >= 16) {
        s.x[0] ^= load64(m, 8); s.x[1] ^= load64(m + 8, 8);
        store64(c, s.x[0], 8); store64(c + 8, s.x[1], 8);
        perm(&s, 8); m += 16; c += 16; mlen -= 16;
    }
    if (mlen >= 8) {
        s.x[0] ^= load64(m, 8);
        s.x[1] ^= load64(m + 8, mlen - 8) ^ pad(mlen - 8);
        store64(c, s.x[0], 8); store64(c + 8, s.x[1], mlen - 8);
    } else {
        s.x[0] ^= load64(m, mlen) ^ pad(mlen);
        store64(c, s.x[0], mlen);
    }
    c += mlen;
    aead_final(&s, k0, k1, c);
    return 0;
}

/* clen includes the 16-byte tag; m must hold clen - 16 bytes. returns 0 or -1 */
int ascon_aead128_decrypt(uint8_t *m, const uint8_t *c, size_t clen,
                          const uint8_t *ad, size_t adlen,
                          const uint8_t *npub, const uint8_t *k) {
    if (clen < 16) return -1;
    state_t s; uint64_t k0, k1;
    size_t mlen = clen - 16;
    uint8_t *m0 = m; size_t mlen0 = mlen;
    aead_init(&s, k, npub, &k0, &k1);
    aead_ad(&s, ad, adlen);
    while (mlen >= 16) {
        uint64_t c0 = load64(c, 8), c1 = load64(c + 8, 8);
        store64(m, s.x[0] ^ c0, 8); store64(m + 8, s.x[1] ^ c1, 8);
        s.x[0] = c0; s.x[1] = c1;
        perm(&s, 8); m += 16; c += 16; mlen -= 16;
    }
    if (mlen >= 8) {
        uint64_t c0 = load64(c, 8), c1 = load64(c + 8, mlen - 8);
        store64(m, s.x[0] ^ c0, 8); store64(m + 8, s.x[1] ^ c1, mlen - 8);
        s.x[0] = c0;
        s.x[1] = (s.x[1] & ~mask(mlen - 8)) ^ c1 ^ pad(mlen - 8);
    } else {
        uint64_t c0 = load64(c, mlen);
        store64(m, s.x[0] ^ c0, mlen);
        s.x[0] = (s.x[0] & ~mask(mlen)) ^ c0 ^ pad(mlen);
    }
    c += mlen;
    uint8_t t[16];
    aead_final(&s, k0, k1, t);
    uint8_t diff = 0;
    for (int i = 0; i < 16; i++) diff |= (uint8_t)(t[i] ^ c[i]);
    if (diff) { memset(m0, 0, mlen0); return -1; }   /* release nothing */
    return 0;
}

/* ---------------- Ascon-XOF128 / Ascon-CXOF128 ---------------- */
#define XOF_IV  0x0000080000cc0003ULL
#define CXOF_IV 0x0000080000cc0004ULL

static void absorb8(state_t *s, const uint8_t *in, size_t len) {
    while (len >= 8) { s->x[0] ^= load64(in, 8); perm(s, 12); in += 8; len -= 8; }
    s->x[0] ^= load64(in, len) ^ pad(len);
    perm(s, 12);
}

static void squeeze8(state_t *s, uint8_t *out, size_t outlen) {
    while (outlen > 8) { store64(out, s->x[0], 8); perm(s, 12); out += 8; outlen -= 8; }
    store64(out, s->x[0], outlen);
}

int ascon_xof128(uint8_t *out, size_t outlen, const uint8_t *in, size_t inlen) {
    state_t s = {{XOF_IV, 0, 0, 0, 0}};
    perm(&s, 12);
    absorb8(&s, in, inlen);
    squeeze8(&s, out, outlen);
    return 0;
}

int ascon_cxof128(uint8_t *out, size_t outlen, const uint8_t *in, size_t inlen,
                  const uint8_t *cs, size_t cslen) {
    if (cslen > 256) return -1;
    state_t s = {{CXOF_IV, 0, 0, 0, 0}};
    perm(&s, 12);
    s.x[0] ^= (uint64_t)cslen * 8;      /* Z0: bit length of customization */
    perm(&s, 12);
    absorb8(&s, cs, cslen);
    absorb8(&s, in, inlen);
    squeeze8(&s, out, outlen);
    return 0;
}

/* ---------------- batch helpers for SHALE stream mode ---------------- */
/* Encrypt nf frames. frames are concatenated in m with offsets moff[i], lengths mlen[i];
 * ciphertexts written to c at coff[i] (each mlen[i]+16 bytes). ad[i] is 32 bytes,
 * nonce[i] is 16 bytes. */
int ascon_aead128_encrypt_batch(uint8_t *c, const int64_t *coff,
                                const uint8_t *m, const int64_t *moff, const int64_t *mlen,
                                const uint8_t *ad, size_t adlen_each,
                                const uint8_t *nonces, const uint8_t *k, int64_t nf) {
    for (int64_t i = 0; i < nf; i++)
        ascon_aead128_encrypt(c + coff[i], m + moff[i], (size_t)mlen[i],
                              ad + i * adlen_each, adlen_each, nonces + 16 * i, k);
    return 0;
}

int ascon_aead128_decrypt_batch(uint8_t *m, const int64_t *moff,
                                const uint8_t *c, const int64_t *coff, const int64_t *clen,
                                const uint8_t *ad, size_t adlen_each,
                                const uint8_t *nonces, const uint8_t *k, int64_t nf,
                                int32_t *ok) {
    int bad = 0;
    for (int64_t i = 0; i < nf; i++) {
        ok[i] = ascon_aead128_decrypt(m + moff[i], c + coff[i], (size_t)clen[i],
                                      ad + i * adlen_each, adlen_each, nonces + 16 * i, k) == 0;
        bad |= !ok[i];
    }
    return bad ? -1 : 0;
}
