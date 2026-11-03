"""Per-packet sizes of real codecs (FFmpeg FLAC, libopus VBR) for CREMA-D and the VCTK profile."""
import sys, os, glob, subprocess, pickle, tempfile, numpy as np
from concurrent.futures import ProcessPoolExecutor

CFG = {
    'flac_20ms': ['-c:a', 'flac', '-compression_level', '5', '-frame_size', '320'],
    'opus_vbr24_20ms': ['-c:a', 'libopus', '-b:a', '24k', '-vbr', 'on', '-frame_duration', '20', '-application', 'voip'],
}

def sizes(f):
    out = {'file': os.path.basename(f)}
    for k, args in CFG.items():
        txt = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', f] + args + ['-f', 'framecrc', '-'],
                             capture_output=True, text=True).stdout
        out[k] = np.array([int(l.split(',')[4]) for l in txt.splitlines() if l and not l.startswith('#')], np.int32)
    return out

if __name__ == '__main__':
    name, pat, lim = sys.argv[1], sys.argv[2], int(sys.argv[3])
    fs = sorted(glob.glob(pat))
    if lim > 0:
        fs = sorted(np.random.default_rng(7).choice(fs, lim, replace=False))
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(sizes, fs, chunksize=20))
    pickle.dump(res, open(f'results/real_{name}.pkl', 'wb'))
    print(name, len(res))
