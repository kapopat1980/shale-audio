import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
fig,ax=plt.subplots(figsize=(7.4,3.9),dpi=300); ax.set_xlim(0,100); ax.set_ylim(0,62); ax.axis('off')
INK='#0b0b0b'; SEC='#52514e'; BLUE='#2a78d6'; ORANGE='#eb6834'; AQUA='#1baf7a'; GRAY='#d9d8d4'
def box(x,y,w,h,title,lines,col):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4,rounding_size=1.2',fc='white',ec=col,lw=1.6))
    ax.add_patch(FancyBboxPatch((x,y+h-5),w,5,boxstyle='round,pad=0.4,rounding_size=1.2',fc=col,ec=col,lw=1.6))
    ax.text(x+w/2,y+h-2.5,title,ha='center',va='center',color='white',fontsize=7.6,fontweight='bold')
    for i,l in enumerate(lines): ax.text(x+1.2,y+h-9.5-i*4.3,l,ha='left',va='center',color=INK,fontsize=6.6)
def arrow(x1,y1,x2,y2,c=SEC):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=10,color=c,lw=1.2))
ax.text(0,55,'PCM audio X\n(N samples, C ch., b bits)',fontsize=6.8,va='center',color=INK)
arrow(10,51,15.5,44)
box(15,10,24,38,'1 · Lossless coder',['frames of F samples','constant / fixed (0-4) / LPC','  order by Levinson error','partitioned Rice residual','L/S, R/S, M/S stereo','verbatim fallback:','  record ≤ F·C·b/8 + 4'],BLUE)
arrow(39.8,29,42.6,29)
box(43,10,22,38,'2 · ECLQ quantiser',['public profile p(l)','DP: min E[pad] + λ·H(B)','K ≤ Kmax buckets','pad record to b_k ≥ l','leakage ≤ log2 K bits','  per record (stream)','  per file (file mode)'],ORANGE)
arrow(65.8,29,68.6,29)
box(69,10,24,38,'3 · Ascon-AEAD128',['K_σ = CXOF128(K ∥ σ)','nonce = (i, final flag)','AD = hdr digest ∥ i ∥ final','128-bit tag per record','constant-time verify,','  reject before release','(NIST SP 800-232)'],AQUA)
ax.text(81,56,'master key K (128 bit)\nrandom salt σ (128 bit)',fontsize=6.8,ha='center',va='center',color=INK)
arrow(81,53,81,48.8)
arrow(93.8,29,96.4,29)
ax.text(96.6,29,'records\nof size\nb_k + 16',fontsize=6.6,va='center',color=INK)
ax.add_patch(FancyBboxPatch((9,0.5),88,5.5,boxstyle='round,pad=0.3,rounding_size=1',fc='#f2f1ee',ec=GRAY,lw=0.8))
ax.text(54,3.25,'Eavesdropper sees: header (N, fs, C, b, F, bucket table, σ)  +  bucket index of every record  —  nothing else',ha='center',va='center',fontsize=6.6,color=INK)
fig.savefig('figures/fig1_pipeline.png',bbox_inches='tight',facecolor='white'); print('ok')
