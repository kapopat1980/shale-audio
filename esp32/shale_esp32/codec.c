/*
 * SHALE lossless coder in C. Same bitstream as shale/codec.py (fixed + LPC prediction,
 * partitioned Rice coding, L/S R/S M/S stereo), written for speed.
 * Compile: gcc -O3 -march=native -ffp-contract=off -fPIC -shared codec.c -lm
 */
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define MAXP 32
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------------- bit writer (MSB first) ---------------- */
typedef struct { uint8_t *buf; size_t pos; uint64_t acc; int nb; } bw_t;

static inline void put(bw_t *w, uint64_t v, int n) {
    if (n == 0) return;
    if (n > 32) { put(w, v >> 32, n - 32); put(w, v & 0xFFFFFFFFULL, 32); return; }
    w->acc = (w->acc << n) | (v & ((n == 64) ? ~0ULL : ((1ULL << n) - 1)));
    w->nb += n;
    while (w->nb >= 8) { w->nb -= 8; w->buf[w->pos++] = (uint8_t)(w->acc >> w->nb); }
    w->acc &= (w->nb ? ((1ULL << w->nb) - 1) : 0);
}
static inline void flushw(bw_t *w) { if (w->nb) put(w, 0, 8 - w->nb); }

/* ---------------- bit reader ---------------- */
typedef struct { const uint8_t *buf; size_t len; uint64_t pos; } br_t;
static inline uint64_t get(br_t *r, int n) {
    uint64_t v = 0;
    while (n > 0) {
        size_t byte = r->pos >> 3; int off = r->pos & 7;
        int take = 8 - off; if (take > n) take = n;
        uint8_t b = byte < r->len ? r->buf[byte] : 0;
        v = (v << take) | ((b >> (8 - off - take)) & ((1u << take) - 1));
        r->pos += take; n -= take;
    }
    return v;
}
static inline uint64_t get_unary(br_t *r) {
    uint64_t q = 0;
    for (;;) {
        size_t byte = r->pos >> 3; int off = r->pos & 7;
        uint8_t b = byte < r->len ? (uint8_t)(r->buf[byte] << off) : 0;
        int avail = 8 - off;
        if (b) { int lz = __builtin_clz((unsigned)b) - 24; r->pos += lz + 1; return q + lz; }
        q += avail; r->pos += avail;
        if (byte >= r->len + 8) return q;   /* corrupt stream guard */
    }
}
static inline uint64_t zig(int64_t r) { return ((uint64_t)r << 1) ^ (uint64_t)(r >> 63); }
static inline int64_t unzig(uint64_t u) { return (int64_t)(u >> 1) ^ -(int64_t)(u & 1); }
static inline int64_t sgn(uint64_t v, int n) { int64_t x = (int64_t)v; if (x >= (1LL << (n - 1))) x -= (1LL << n); return x; }

/* ---------------- prediction ---------------- */
static void fixed_res(const int64_t *x, int n, int order, int64_t *o) {
    for (int i = 0; i < order && i < n; i++) o[i] = 0;
    for (int i = order; i < n; i++) {
        switch (order) {
        case 0: o[i] = x[i]; break;
        case 1: o[i] = x[i] - x[i-1]; break;
        case 2: o[i] = x[i] - 2*x[i-1] + x[i-2]; break;
        case 3: o[i] = x[i] - 3*x[i-1] + 3*x[i-2] - x[i-3]; break;
        default: o[i] = x[i] - 4*x[i-1] + 6*x[i-2] - 4*x[i-3] + x[i-4];
        }
    }
}
static void lpc_res(const int64_t *x, int n, const int64_t *c, int order, int shift, int64_t *o) {
    for (int i = 0; i < order; i++) o[i] = 0;
    for (int i = order; i < n; i++) {
        int64_t s = 0;
        for (int j = 0; j < order; j++) s += c[j] * x[i-1-j];
        o[i] = x[i] - (s >> shift);
    }
}
static void levinson(const double *r, int maxp, double lpc[MAXP+1][MAXP+1], double *errs) {
    double err = r[0], a[MAXP+1] = {0}, tmp[MAXP+1];
    errs[0] = err;
    for (int p = 1; p <= maxp; p++) {
        if (err <= 0) { for (int q = p; q <= maxp; q++) { for (int j = 0; j < q; j++) lpc[q-1][j] = 0; errs[q] = 0; } return; }
        double acc = r[p];
        for (int j = 1; j < p; j++) acc -= a[j] * r[p-j];
        double k = acc / err;
        for (int j = 0; j < p; j++) tmp[j] = a[j];
        a[p] = k;
        for (int j = 1; j < p; j++) a[j] = tmp[j] - k * tmp[p-j];
        err *= (1.0 - k * k);
        for (int j = 0; j < p; j++) lpc[p-1][j] = a[j+1];
        errs[p] = err;
    }
}
static int quantise(const double *a, int order, int prec, int64_t *qc) {
    double cmax = 0;
    for (int j = 0; j < order; j++) if (fabs(a[j]) > cmax) cmax = fabs(a[j]);
    if (cmax <= 0) return -1;
    int e = (int)floor(log2(cmax)) + 1;
    int shift = prec - 1 - e;
    if (shift > 20) shift = 20;
    if (shift < 0) return -1;
    int64_t lim = (1LL << (prec - 1)) - 1;
    double err = 0;
    for (int j = 0; j < order; j++) {
        err += a[j] * (double)(1LL << shift);
        int64_t q = (int64_t)nearbyint(err);
        if (q > lim) q = lim;
        if (q < -lim - 1) q = -lim - 1;
        qc[j] = q; err -= (double)q;
    }
    return shift;
}

/* ---------------- Rice ---------------- */
static int rice_k(uint64_t s, int64_t n) {
    if (n <= 0 || s == 0) return 0;
    double m = (double)s / (double)n;
    int k = 0;
    while (k < 30 && (double)(1ULL << (k + 1)) <= m + 1) k++;
    return k;
}
static int64_t best_partition(const int64_t *res, int order, int maxpo, int n, int *bpo, uint64_t *ps) {
    ps[0] = 0;
    for (int i = 0; i < n; i++) ps[i+1] = ps[i] + (i < order ? 0 : zig(res[i]));
    int64_t best = 1LL << 62; *bpo = 0;
    for (int po = 0; po <= maxpo; po++) {
        int np_ = 1 << po;
        if (n % np_) break;
        int plen = n / np_;
        if (plen <= order) break;
        int64_t bits = 4;
        for (int p = 0; p < np_; p++) {
            int a = p * plen, b = a + plen;
            int64_t cnt = p == 0 ? plen - order : plen;
            uint64_t s = ps[b] - ps[a];
            int k = rice_k(s, cnt);
            bits += 5 + cnt * (k + 1) + (int64_t)(s >> k);
        }
        if (bits < best) { best = bits; *bpo = po; }
    }
    return best;
}
static void write_residual(bw_t *w, const int64_t *res, int order, int po, int n) {
    put(w, po, 4);
    int np_ = 1 << po, plen = n / np_;
    for (int p = 0; p < np_; p++) {
        int a = p * plen + (p == 0 ? order : 0), b = (p + 1) * plen;
        uint64_t s = 0, mx = 0;
        for (int i = a; i < b; i++) { uint64_t u = zig(res[i]); s += u; if (u > mx) mx = u; }
        int k = rice_k(s, b - a), bestk = k;
        int64_t bestc = 1LL << 62;
        for (int kk = (k - 1 > 0 ? k - 1 : 0); kk <= (k + 1 < 30 ? k + 1 : 30); kk++) {
            int64_t c = 0;
            for (int i = a; i < b; i++) c += (int64_t)(zig(res[i]) >> kk) + kk + 1;
            if (c < bestc) { bestc = c; bestk = kk; }
        }
        int wd = 0;
        while (wd < 40 && (1ULL << wd) <= mx) wd++;
        if ((int64_t)wd * (b - a) + 6 < bestc) {
            put(w, 31, 5); put(w, wd, 6);
            for (int i = a; i < b; i++) put(w, zig(res[i]), wd);
        } else {
            put(w, bestk, 5);
            for (int i = a; i < b; i++) {
                uint64_t u = zig(res[i]), q = u >> bestk;
                while (q >= 32) { put(w, 0, 32); q -= 32; }
                put(w, 1, (int)q + 1);
                put(w, u & ((1ULL << bestk) - 1), bestk);
            }
        }
    }
}
static void read_residual(br_t *r, int64_t *res, int order, int n) {
    int po = (int)get(r, 4), np_ = 1 << po, plen = n / np_;
    for (int p = 0; p < np_; p++) {
        int a = p * plen + (p == 0 ? order : 0), b = (p + 1) * plen;
        int k = (int)get(r, 5);
        if (k == 31) { int wd = (int)get(r, 6); for (int i = a; i < b; i++) res[i] = unzig(get(r, wd)); }
        else for (int i = a; i < b; i++) { uint64_t q = get_unary(r); uint64_t lo = get(r, k); res[i] = unzig((q << k) | lo); }
    }
}

/* ---------------- subframe ---------------- */
typedef struct { int64_t *res, *bres; uint64_t *ps; double *xw, *win; } work_t;

static int64_t encode_sub(bw_t *w, const int64_t *x, int n, int bits, int maxp, int prec, int maxpo, work_t *wk, int dry) {
    int cst = 1;
    for (int i = 1; i < n; i++) if (x[i] != x[0]) { cst = 0; break; }
    uint64_t mask = (1ULL << bits) - 1;
    if (cst) { if (!dry) { put(w, 1, 3); put(w, (uint64_t)x[0] & mask, bits); } return 3 + bits; }
    int64_t best = 3 + (int64_t)bits * n; int btype = 0, border = 0, bpo = 0, bshift = 0, po;
    double e[5] = {0};
    for (int i = 4; i < n; i++) {
        int64_t d0 = x[i], d1 = d0 - x[i-1], d2 = d1 - (x[i-1] - x[i-2]);
        int64_t d3 = d2 - (x[i-1] - 2*x[i-2] + x[i-3]);
        int64_t d4 = d3 - (x[i-1] - 3*x[i-2] + 3*x[i-3] - x[i-4]);
        e[0] += llabs(d0); e[1] += llabs(d1); e[2] += llabs(d2); e[3] += llabs(d3); e[4] += llabs(d4);
    }
    int fo = 0; for (int o = 1; o < 5; o++) if (e[o] < e[fo]) fo = o;
    int olo = fo, ohi = fo; if (n <= 4) { olo = 0; ohi = n < 5 ? n - 1 : 4; }
    for (int order = olo; order <= ohi; order++) {
        fixed_res(x, n, order, wk->res);
        int64_t c = best_partition(wk->res, order, maxpo, n, &po, wk->ps) + 3 + 3 + order * bits;
        if (c < best) { best = c; btype = 2; border = order; bpo = po; memcpy(wk->bres, wk->res, n * sizeof(int64_t)); }
    }
    int64_t bestc[MAXP]; int pmax = maxp < n - 1 ? maxp : n - 1;
    if (pmax >= 1) {
        static double r[MAXP+1], errs[MAXP+1], lpc[MAXP+1][MAXP+1];   /* static: keeps the MCU stack small */
        for (int i = 0; i < n; i++) wk->xw[i] = (double)x[i] * wk->win[i];
        for (int lag = 0; lag <= pmax; lag++) { double s = 0; for (int i = lag; i < n; i++) s += wk->xw[i] * wk->xw[i-lag]; r[lag] = s; }
        if (r[0] > 0) {
            r[0] *= 1.0 + 1e-9;
            levinson(r, pmax, lpc, errs);
            double be = 1e300; int bo = 1;
            for (int o = 1; o <= pmax; o++) {
                double ev = errs[o] / n, bps = ev > 1e-12 ? 0.5 * log2(ev) : 0.0;
                if (bps < 0) bps = 0;
                double est = n * bps + o * (prec + bits);
                if (est < be) { be = est; bo = o; }
            }
            int64_t qc[MAXP];
            int shift = quantise(lpc[bo-1], bo, prec, qc);
            if (shift >= 0) {
                lpc_res(x, n, qc, bo, shift, wk->res);
                int64_t c = best_partition(wk->res, bo, maxpo, n, &po, wk->ps) + 3 + 5 + 4 + 5 + bo * prec + bo * bits;
                if (c < best) { best = c; btype = 3; border = bo; bpo = po; bshift = shift; memcpy(wk->bres, wk->res, n * sizeof(int64_t)); memcpy(bestc, qc, bo * sizeof(int64_t)); }
            }
        }
    }
    if (dry) return best;
    if (btype == 0) { put(w, 0, 3); for (int i = 0; i < n; i++) put(w, (uint64_t)x[i] & mask, bits); }
    else if (btype == 2) {
        put(w, 2, 3); put(w, border, 3);
        for (int i = 0; i < border; i++) put(w, (uint64_t)x[i] & mask, bits);
        write_residual(w, wk->bres, border, bpo, n);
    } else {
        put(w, 3, 3); put(w, border, 5); put(w, prec - 1, 4); put(w, bshift, 5);
        uint64_t pm = (1ULL << prec) - 1;
        for (int j = 0; j < border; j++) put(w, (uint64_t)bestc[j] & pm, prec);
        for (int i = 0; i < border; i++) put(w, (uint64_t)x[i] & mask, bits);
        write_residual(w, wk->bres, border, bpo, n);
    }
    return best;
}

static void decode_sub(br_t *r, int64_t *x, int n, int bits, int64_t *res) {
    int t = (int)get(r, 3);
    if (t == 1) { int64_t v = sgn(get(r, bits), bits); for (int i = 0; i < n; i++) x[i] = v; }
    else if (t == 0) { for (int i = 0; i < n; i++) x[i] = sgn(get(r, bits), bits); }
    else if (t == 2) {
        int order = (int)get(r, 3);
        for (int i = 0; i < order; i++) x[i] = sgn(get(r, bits), bits);
        read_residual(r, res, order, n);
        for (int i = order; i < n; i++) {
            switch (order) {
            case 0: x[i] = res[i]; break;
            case 1: x[i] = res[i] + x[i-1]; break;
            case 2: x[i] = res[i] + 2*x[i-1] - x[i-2]; break;
            case 3: x[i] = res[i] + 3*x[i-1] - 3*x[i-2] + x[i-3]; break;
            default: x[i] = res[i] + 4*x[i-1] - 6*x[i-2] + 4*x[i-3] - x[i-4];
            }
        }
    } else {
        int order = (int)get(r, 5), prec = (int)get(r, 4) + 1, shift = (int)get(r, 5);
        int64_t c[MAXP];
        for (int j = 0; j < order; j++) c[j] = sgn(get(r, prec), prec);
        for (int i = 0; i < order; i++) x[i] = sgn(get(r, bits), bits);
        read_residual(r, res, order, n);
        for (int i = order; i < n; i++) { int64_t s = 0; for (int j = 0; j < order; j++) s += c[j] * x[i-1-j]; x[i] = res[i] + (s >> shift); }
    }
}

static void tukey(double *w, int n) {
    for (int i = 0; i < n; i++) w[i] = 1.0;
    double a = 0.5; int m = (int)(a * (n - 1) / 2);
    for (int i = 0; i <= m; i++) {
        double v = n > 1 ? 0.5 * (1 - cos(2 * M_PI * i / (a * (n - 1)))) : 1.0;
        w[i] = v; w[n-1-i] = v;
    }
}

/* encode one frame: x interleaved (n x C). returns number of bytes */
static int encode_frame(const int64_t *x, int n, int C, int bits, int maxp, int prec, int maxpo, uint8_t *out, work_t *wk, int64_t *ch) {
    bw_t w = {out, 0, 0, 0};
    tukey(wk->win, n);
    if (C == 2) {
        int64_t *L = ch, *R = ch + n, *S = ch + 2*n, *M = ch + 3*n;
        for (int i = 0; i < n; i++) { L[i] = x[2*i]; R[i] = x[2*i+1]; S[i] = L[i] - R[i]; M[i] = (L[i] + R[i]) >> 1; }
        int64_t cL = encode_sub(&w, L, n, bits, maxp, prec, maxpo, wk, 1), cR = encode_sub(&w, R, n, bits, maxp, prec, maxpo, wk, 1);
        int64_t cS = encode_sub(&w, S, n, bits + 1, maxp, prec, maxpo, wk, 1), cM = encode_sub(&w, M, n, bits, maxp, prec, maxpo, wk, 1);
        int64_t cs[4] = {cL + cR, cL + cS, cR + cS, cM + cS}; int mode = 0;
        for (int k = 1; k < 4; k++) if (cs[k] < cs[mode]) mode = k;
        put(&w, mode, 2);
        const int64_t *A = mode == 0 ? L : mode == 1 ? L : mode == 2 ? R : M;
        const int64_t *B = mode == 0 ? R : S;
        encode_sub(&w, A, n, bits, maxp, prec, maxpo, wk, 0);
        encode_sub(&w, B, n, mode == 0 ? bits : bits + 1, maxp, prec, maxpo, wk, 0);
    } else {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < n; i++) ch[i] = x[i*C + c];
            encode_sub(&w, ch, n, bits, maxp, prec, maxpo, wk, 0);
        }
    }
    flushw(&w);
    return (int)w.pos;
}

/* Encode N x C interleaved samples in frames of F. out must hold nframes*maxbytes;
 * frame i written at out + i*maxbytes, length in lens[i]. Returns number of frames. */
int shale_encode_frames(const int64_t *x, int64_t N, int C, int bits, int F, int maxp, int prec, int maxpo,
                        uint8_t *out, int64_t maxbytes, int64_t *lens) {
    work_t wk;
    wk.res = malloc(F * sizeof(int64_t)); wk.bres = malloc(F * sizeof(int64_t));
    wk.ps = malloc((F + 1) * sizeof(uint64_t)); wk.xw = malloc(F * sizeof(double)); wk.win = malloc(F * sizeof(double));
    int64_t *ch = malloc(4 * (int64_t)F * sizeof(int64_t));
    int64_t nf = 0;
    for (int64_t s = 0; s < N; s += F, nf++) {
        int n = (int)((N - s) < F ? (N - s) : F);
        lens[nf] = encode_frame(x + s * C, n, C, bits, maxp, prec, maxpo, out + nf * maxbytes, &wk, ch);
    }
    free(wk.res); free(wk.bres); free(wk.ps); free(wk.xw); free(wk.win); free(ch);
    return (int)nf;
}

/* Decode one frame of n samples x C channels (interleaved output). */
int shale_decode_frame(const uint8_t *buf, int64_t len, int n, int C, int bits, int64_t *out) {
    br_t r = {buf, (size_t)len, 0};
    int64_t *a = malloc(n * sizeof(int64_t)), *b = malloc(n * sizeof(int64_t)), *res = malloc(n * sizeof(int64_t));
    if (C == 2) {
        int mode = (int)get(&r, 2);
        decode_sub(&r, a, n, bits, res);
        decode_sub(&r, b, n, mode == 0 ? bits : bits + 1, res);
        for (int i = 0; i < n; i++) {
            int64_t L, R;
            if (mode == 0) { L = a[i]; R = b[i]; }
            else if (mode == 1) { L = a[i]; R = a[i] - b[i]; }
            else if (mode == 2) { R = a[i]; L = a[i] + b[i]; }
            else { int64_t m = (a[i] << 1) | (b[i] & 1); L = (m + b[i]) >> 1; R = (m - b[i]) >> 1; }
            out[2*i] = L; out[2*i+1] = R;
        }
    } else {
        for (int c = 0; c < C; c++) { decode_sub(&r, a, n, bits, res); for (int i = 0; i < n; i++) out[i*C + c] = a[i]; }
    }
    free(a); free(b); free(res);
    return (int)((r.pos + 7) >> 3);
}
