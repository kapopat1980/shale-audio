"""Design the general-purpose file-mode buckets (K = 2, 4, 8) on the public profile:
VCTK speech + MS-SNSD training noise, equally weighted. Writes results/file_ratios.json."""
import sys, pickle, json, numpy as np
sys.path.insert(0, '.')
from shale import eclq, core
R = 'results'
rat = lambda d: (8 + d[4096].sum()) / core.file_top(d['N'], d['C'], d.get('bits', 16), 4096)
v = pickle.load(open(f'{R}/len_vctk.pkl', 'rb')); n = pickle.load(open(f'{R}/len_noisetrain.pkl', 'rb'))
rv = np.array([rat(d) for d in v]); rn = np.array([rat(d) for d in n])
w = np.concatenate([np.full(len(rv), 1 / len(rv)), np.full(len(rn), 1 / len(rn))])
lv = np.ceil(np.concatenate([rv, rn]) * 10000).astype(int)
out = {f'K{K}': (eclq.design(lv, 10000, K, 0.0, 10, weights=w) / 10000).tolist() for K in (2, 4, 8)}
json.dump(out, open(f'{R}/file_ratios.json', 'w')); print(out)
