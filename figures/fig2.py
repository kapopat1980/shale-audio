import sys, pickle, numpy as np, soundfile as sf
sys.path.insert(0,'.')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from shale import eclq, core
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.edgecolor':'#9a9994','axes.linewidth':0.6,
  'xtick.color':'#52514e','ytick.color':'#52514e','axes.labelcolor':'#0b0b0b'})
BLUE='#2a78d6'; ORANGE='#eb6834'; GRAY='#b9b8b3'
v=pickle.load(open('results/len_vctk.pkl','rb'))
c={d['file']:d for d in pickle.load(open('results/len_crema.pkl','rb'))}
F=320; top=core.frame_top(F,1,16)
b4=eclq.design(np.concatenate([d[F] for d in v]),top,4,0.0,4)
files=['1001_DFA_ANG_XX.wav','1001_DFA_SAD_XX.wav','1001_IEO_ANG_HI.wav']
labels=['"Don\'t forget a jacket", angry','"Don\'t forget a jacket", sad','"It\'s eleven o\'clock", angry']
fig,axs=plt.subplots(3,3,figsize=(7.2,4.6),dpi=300,sharex='col')
for j,(f,lab) in enumerate(zip(files,labels)):
    x,sr=sf.read('data/CREMA-D/AudioWAV/'+f,dtype='int16'); t=np.arange(len(x))/sr
    L=c[f][F]; tr=np.arange(len(L))*F/sr
    ax=axs[0,j]; ax.plot(t,x/32768,color=GRAY,lw=0.3); ax.set_ylim(-1,1); ax.set_title(lab,fontsize=7.5,color='#0b0b0b')
    ax=axs[1,j]; ax.step(tr,L+16,where='post',color=ORANGE,lw=1); ax.set_ylim(0,top+40)
    ax=axs[2,j]; ax.step(tr,b4[np.searchsorted(b4,L)]+16,where='post',color=BLUE,lw=1); ax.set_ylim(0,top+40)
    axs[2,j].set_xlabel('time (s)')
    for a in axs[:,j]:
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(axis='y',color='#e6e5e1',lw=0.5)
axs[0,0].set_ylabel('PCM\n(normalised)'); axs[1,0].set_ylabel('record size (B)\nno padding'); axs[2,0].set_ylabel('record size (B)\nSHALE, K = 4')
for j in (1,2):
    for i in range(3): axs[i,j].set_yticklabels([])
fig.align_ylabels(axs[:,0]); fig.tight_layout(h_pad=0.6,w_pad=0.6)
fig.savefig('figures/fig2_trace.png',bbox_inches='tight',facecolor='white'); print(b4)
