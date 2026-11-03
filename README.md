# SHALE — Size-Hiding Authenticated Lossless audio Encryption

Reference implementation and complete experiment code for the paper

> L. Patil, K. Popat, *Compressed Audio Encryption Leaks What Was Said: Size-Hiding Authenticated Lossless Encryption with Entropy-Constrained Length Quantisation* (submitted, 2026).

Compressing audio losslessly before authenticated encryption halves its size, but the ciphertext **sizes** then reveal what was said. SHALE combines:

1. a frame-independent lossless coder (fixed/LPC prediction, partitioned Rice coding, stereo decorrelation; Numba and C implementations with identical bitstreams),
2. **ECLQ**, an entropy-constrained length quantiser that pads every record to one of K public bucket sizes chosen by an exact dynamic program, so sizes reveal at most log2 K bits per record (stream mode) or per file (file mode) beyond the duration, and
3. **Ascon-AEAD128** (NIST SP 800-232) with Ascon-CXOF128 key derivation, plus a nonce-misuse-resistant **SHALE-SIV** file mode.

Decryption is bit-exact. Every record, the header, record order and truncation are authenticated.

## Quick start

```bash
pip install -r requirements.txt
make            # builds lib/libshale_ascon.so, lib/libshale_codec.so and ./shale_bench
make test       # 3,203 official Ascon KATs + data-free round-trip and tamper tests
```

```python
import os, numpy as np, soundfile as sf
from shale import core
key = os.urandom(16)
x, sr = sf.read('speech.wav', dtype='int16')               # 16-bit PCM (int32 for 24-bit)
ratios = [0.525, 0.63, 0.762, 1.0]                         # file buckets (results/file_ratios.json, K4)
blob, bucket = core.encrypt_file(key, x.astype(np.int64), sr, 16, 4096, ratios)
y = core.decrypt_file(key, blob, ratios)                    # raises core.AuthError on any tampering
assert np.array_equal(y[:, 0], x)
core.USE_C = True                                           # optional: use the faster C coder
```

Stream mode (one AEAD record per frame, e.g. 20 ms packets): `core.encrypt_stream` / `core.decrypt_stream`.
Misuse-resistant files: `core.encrypt_file_siv` / `core.decrypt_file_siv`.
Bucket design: `shale.eclq.design(lengths, top, K)` (minimum overhead) and `shale.eclq.design_entropy(lengths, top, Kmax, H)` (entropy-constrained).

## Repository layout

| Path | Contents |
|---|---|
| `shale/` | Python package: `codec.py` (Numba coder), `ccodec.py` (C coder binding), `eclq.py`, `ascon.py`, `core.py` (stream, file and SIV modes) |
| `csrc/` | C sources: Ascon-AEAD128/XOF128/CXOF128 written from SP 800-232, the lossless coder, a portable benchmark |
| `tests/` | official Ascon KAT files and tests |
| `exp/` | every experiment in the paper (see below) |
| `kaggle/` | GPU deep-learning attack notebook (CNN + Transformer; CREMA-D and Speech Commands) |
| `esp32/` | Arduino sketches running the full SHALE pipeline: `shale_esp32/` (ESP32) and `shale_esp8266/` (NodeMCU ESP8266, used in the paper) |
| `figures/` | figure scripts and the figures |
| `results/` | all result files (JSON) reported in the paper |

## Data

Nothing is redistributed here. Place the public corpora under `data/`:

```bash
mkdir -p data && cd data
GIT_LFS_SKIP_SMUDGE=0 git clone https://github.com/CheyneyComputerScience/CREMA-D.git   # needs git-lfs (AudioWAV/, VideoDemographics.csv)
git clone https://github.com/microsoft/MS-SNSD.git                                    # clean_train/ (VCTK), noise_train/, noise_test/
git clone https://github.com/flippy-fyp/Bach10_v1.1.git Bach10
cd .. && python exp/make_bach.py          # stereo mixes -> data/bach16, data/bach24
```

The Kaggle notebook uses the Kaggle copies `ejlok1/cremad` and `yashdogra/speech-commands`.

## Reproducing the paper

Run everything from the repository root. Set `export OMP_NUM_THREADS=1` on small machines (scikit-learn thread contention).

| Step | Command | Output (results/) | Paper |
|---|---|---|---|
| Record-length profiles | `python exp/lengths.py crema 'data/CREMA-D/AudioWAV/*.wav' 0`<br>`python exp/lengths.py vctk 'data/MS-SNSD/clean_train/*.wav' 4000`<br>`python exp/lengths.py noisetrain 'data/MS-SNSD/noise_train/*.wav' 0`<br>`python exp/lengths.py noisetest 'data/MS-SNSD/noise_test/*.wav' 0`<br>`python exp/lengths.py bach16 'data/bach16/*.wav' 0`<br>`python exp/lengths.py bach24 'data/bach24/*.wav' 0` | `len_*.pkl` | all |
| General file buckets | `python exp/file_ratios.py` | `file_ratios.json` | §4.2 |
| Length attack (duration, file, stream) | `python exp/attack.py` | `attack.json` | Tables 4, 6, 7, 10 |
| File mode, entropy-constrained | `python exp/attack_file_extra.py` | `attack_file_extra.json` | Table 10 |
| In-domain profile | `python exp/attack_indomain.py` | `attack_indomain.json` | Table 10 |
| Stream mode, entropy-constrained | `python exp/attack_stream_extra.py 320 vctk 0.1,0.3` (also 1600; `indomain`) | `attack_stream_*.json` | Tables 6–7 |
| Padmé baseline | `python exp/attack_padme.py` | `attack_padme.json` | Tables 5–7, 10 |
| Deployed codecs (FLAC, Opus) | `python exp/realcodec_shard.py crema 'data/CREMA-D/AudioWAV/*.wav' 0 0 2` (shard 0 and 1; same for `vctk ... 4000`), `python exp/realcodec_merge.py`, `python exp/attack_real.py opus_vbr24_20ms,flac_20ms` | `attack_real.json` | Table 5 |
| Paired significance | `python exp/significance.py` | `significance.json` | Table 11 |
| Deep-learning attack | `kaggle/SHALE_deep_attack_kaggle.ipynb` on a Kaggle GPU | `shale_dl_results.json` | Tables 8–9 |
| Compression vs FLAC | `python exp/sizes.py` | `sizes.json` | Table 3 |
| Throughput | `python exp/bench.py`, `python exp/bench_c.py` | `bench.json`, `ccodec_bench.json` | Table 12 |
| Statistics, NIST, sensitivity, tamper, loss | `python exp/stats.py` | `stats.json` | Tables 14–16 |
| SIV mode | `python exp/siv_test.py` | `siv.json` | Table 16 |
| ESP8266 measurement | `esp32/shale_esp8266/` on a NodeMCU 1.0 (Arduino IDE, 160 MHz) | `esp8266.json` | Table 13 |
| Bit-exact check of every file | `python exp/roundtrip_all.py` | `roundtrip_all.json` | §6.1 |
| Figures | `python figures/fig1.py` … `fig4.py` | `figures/*.png` | Figs 1–4 |

Table numbers refer to the submitted manuscript.

## Embedded benchmark

`./shale_bench file.wav 4096 5` runs the full C pipeline (coder, padding, Ascon, decrypt and verify) on any Linux board. `esp32/shale_esp32/` runs it on an ESP32 from the Arduino IDE (see its README); `esp32/shale_esp8266/` is the exact sketch used for the paper's NodeMCU 1.0 (ESP8266) measurement (`results/esp8266.json`, Table 13).

## Citation

If you use this code, please cite the paper (citation will be added on publication).

## Licence

MIT (see `LICENSE`). The Ascon KAT files come from the official `ascon-c` repository (CC0).
