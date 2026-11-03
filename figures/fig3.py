import json, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = 'results'
J = lambda f: json.load(open(f'{R}/{f}'))
att = J('attack.json'); PD = J('attack_padme.json'); fx = J('attack_file_extra.json'); ind = J('attack_indomain.json')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': '#9a9994', 'axes.linewidth': 0.6,
                     'xtick.color': '#52514e', 'ytick.color': '#52514e'})
BLUE, ORANGE, AQUA, VIOLET, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#8a8984'
best = lambda r, t: max(r['tasks'][t]['hgb'][2], r['tasks'][t]['rocket'][2])
dur = {t: v[2] for t, v in att['duration'].items()}


def series(task):
    S = {}
    for F, lab, col, mk in ((320, 'stream 20 ms', ORANGE, 'o'), (1600, 'stream 100 ms', AQUA, 's')):
        pts = []
        for k, r in att[f'stream_{F}'].items():
            pts.append(((r['mean_padded'] + 16) / (r['top'] - 4), best(r, task), k))
        for pk in ('vctk',):
            p = f'{R}/attack_stream_{F}_{pk}.json'
            if os.path.exists(p):
                for k, r in json.load(open(p)).items():
                    pts.append((r['size'], best(r, task), k))
        pd = PD[f'stream_{F}']; pts.append((pd['size'], best(pd, task), 'padme'))
        S[lab] = (pts, col, mk)
    pts = [(PD['file']['size'], PD['file']['tasks'][task][2], 'padme'), (att['file']['none']['mean_size'] / att['file']['none']['mean_raw'], att['file']['none']['tasks'][task][2], 'none')]
    for k in ('eclq_K2', 'eclq_K4', 'eclq_K8'):
        r = att['file'][k]; pts.append((r['mean_size'] / r['mean_raw'], r['tasks'][task][2], k))
    for k, r in fx.items():
        if k.startswith('H'): pts.append((r['mean_size'] / r['mean_raw'], r['tasks'][task][2], k))
    S['file, public profile'] = (pts, BLUE, 'D')
    pts = [(r['size'], r['tasks'][task][2], k) for k, r in ind['file'].items() if k not in ('duration', 'K1')]
    S['file, in-domain profile*'] = (pts, VIOLET, '^')
    return S


fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), dpi=300)
for ax, task, title in ((axs[0], 'emotion', 'Emotion (6 classes)'), (axs[1], 'sentence', 'Sentence (12 classes)')):
    for lab, (pts, col, mk) in series(task).items():
        pts = sorted(pts)
        ax.scatter([p[0] for p in pts], [100 * p[1] for p in pts], s=22, marker=mk, color=col, edgecolor='white', linewidth=0.6, label=lab, zorder=3)
    ax.axhline(100 * dur[task], color=GRAY, ls='--', lw=1)
    ax.scatter([1.0], [100 * dur[task]], s=34, marker='*', color='#0b0b0b', zorder=4, label='PCM ciphers (duration only)')
    ax.text(0.505, 100 * dur[task] + 0.8, 'duration-only baseline', color='#52514e', fontsize=6.5)
    ax.set_title(title, fontsize=8.5); ax.set_xlabel('ciphertext size / PCM size'); ax.set_xlim(0.48, 1.04)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.grid(color='#e6e5e1', lw=0.5)
axs[0].set_ylabel('attack accuracy (%)')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=5, frameon=False, fontsize=6.6, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig('figures/fig3_pareto.png', bbox_inches='tight', facecolor='white')
print('ok')
