import sys, glob, pickle, numpy as np
sys.argv_saved = list(sys.argv); sys.argv = ['x']
exec(open('exp/realcodec.py').read().split('if __name__')[0])
name, pat, lim, k, K = sys.argv_saved[1], sys.argv_saved[2], int(sys.argv_saved[3]), int(sys.argv_saved[4]), int(sys.argv_saved[5])
fs = sorted(glob.glob(pat))
if lim > 0: fs = sorted(np.random.default_rng(7).choice(fs, lim, replace=False))
fs = fs[k::K]
res = []
for i, f in enumerate(fs):
    res.append(sizes(f))
    if i % 500 == 0: print(name, k, i, len(fs), flush=True)
pickle.dump(res, open(f'results/real_{name}_{k}.pkl', 'wb'))
print('done', name, k, flush=True)
