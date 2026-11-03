import sys, pickle, json, numpy as np
sys.path.insert(0,'.')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from shale import eclq, core
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.edgecolor':'#9a9994','axes.linewidth':0.6,'xtick.color':'#52514e','ytick.color':'#52514e'})
BLUE='#2a78d6'; ORANGE='#eb6834'; AQUA='#1baf7a'; GRAY='#b9b8b3'
v=pickle.load(open('results/len_vctk.pkl','rb')); c=pickle.load(open('results/len_crema.pkl','rb'))
F=320; top=core.frame_top(F,1,16); prof=np.concatenate([d[F] for d in v]); cr=np.concatenate([d[F] for d in c])
designs=[('min. overhead, K = 4 (λ = 0)',eclq.design(prof,top,4,0.0,4),ORANGE),('entropy-constrained, H ≤ 0.1 bit',eclq.design_entropy(prof,top,4,0.1,4),BLUE)]
fig,ax=plt.subplots(figsize=(7.2,2.7),dpi=300)
bins=np.arange(0,top+8,8)
ax.hist(prof,bins=bins,density=True,color=GRAY,alpha=0.9,label='public profile (VCTK)',edgecolor='white',linewidth=0.2)
ax.hist(cr,bins=bins,density=True,histtype='step',color='#52514e',lw=0.9,label='attacked data (CREMA-D)')
ymax=ax.get_ylim()[1]
for k,(lab,b,col) in enumerate(designs):
    s=eclq.stats(prof,b)
    for j,t in enumerate(b):
        ax.axvline(t,color=col,lw=1.3,ls='-' if k==0 else '--',label=f'{lab}: size {s["mean_padded"]/top:.2f}, H = {s["H"]:.2f} bit' if j==0 else None)
ax.set_xlabel('record length before padding (bytes), 20 ms frames, 16 kHz mono'); ax.set_ylabel('density')
for sp in ('top','right'): ax.spines[sp].set_visible(False)
ax.legend(frameon=True,facecolor='white',edgecolor='none',framealpha=0.95,fontsize=6.6,loc='upper right'); ax.set_xlim(0,top+10)
fig.tight_layout(); fig.savefig('figures/fig4_buckets.png',bbox_inches='tight',facecolor='white')
print([d[1].tolist() for d in designs])
