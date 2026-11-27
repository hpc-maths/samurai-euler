# Trace les cellules et les fantomes ecrits par --save-ghosts.
# Usage : python plot_cells_ghosts.py <fichier>_ghosts_0000.vtu <sortie>.png "<titre>"
import sys
import re, struct, numpy as np
def read_vtu(f):
    b=open(f,'rb').read(); i=b.index(b'<AppendedData'); head=b[:i].decode(); data=b[b.index(b'_',i)+1:]
    arrs={}; k=0
    for m in re.finditer(r'<DataArray type="(\w+)"(?: Name="(\w+)")?(?: NumberOfComponents="(\d)")?[^>]*offset="(\d+)"',head):
        t,name,nc,off=m.groups(); off=int(off)
        n=struct.unpack('<Q',data[off:off+8])[0]
        dt={'Float32':np.float32,'Int32':np.int32,'UInt8':np.uint8}[t]
        a=np.frombuffer(data[off+8:off+8+n],dtype=dt)
        if name is None: name='points'; a=a.reshape(-1,3)
        arrs[name]=a
    ends=arrs['offsets']; starts=np.concatenate([[0],ends[:-1]])
    arrs['polys']=[arrs['points'][s:e,:2] for s,e in zip(starts,ends)]
    return arrs


import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
COL={0:'#cfe3ee',1:'#f4b183',2:'#d9734e'}
def draw(ax,a,c0=None,R=None,lw=0.4,front=None):
    order=np.argsort(a['ghost'])  # cells first, then ghosts on top
    for k in order:
        p=a['polys'][k]; c=p.mean(0)
        if c0 is not None and np.hypot(c[0]-c0[0],c[1]-c0[1])>R*1.7: continue
        g=a['ghost'][k]; l=min(int(a['layer'][k]),2)
        ax.add_patch(Polygon(p,closed=True,fc=COL[l],alpha=0.55 if g else 1.0,ec='#2b2b2b',lw=lw))
    th=np.linspace(0,2*np.pi,600)
    for r in (1,2): ax.plot(r*np.cos(th),r*np.sin(th),color='#1f5fa8',ls='--',lw=1.0)
    if c0 is not None: ax.set_xlim(c0[0]-R,c0[0]+R); ax.set_ylim(c0[1]-R,c0[1]+R)
    else: ax.set_xlim(-2.3,2.3); ax.set_ylim(-2.3,2.3)
    ax.set_aspect('equal'); ax.tick_params(labelsize=8)
leg=[Patch(fc=COL[0],ec='#2b2b2b',label='cellules'),Patch(fc=COL[1],ec='#2b2b2b',alpha=0.55,label='fantômes, couche 1'),
     Patch(fc=COL[2],ec='#2b2b2b',alpha=0.55,label='fantômes, couche 2'),plt.Line2D([],[],color='#1f5fa8',ls='--',label='cercles exacts')]
f, out, title = sys.argv[1], sys.argv[2], sys.argv[3]
a=read_vtu(f)
fig=plt.figure(figsize=(15,10))
ax0=fig.add_axes([0.02,0.06,0.48,0.86]); draw(ax0,a,lw=0.25); ax0.set_title('vue d\'ensemble')
for r in [((1.414,1.414),0.22),((0.707,0.707),0.22),((1.93,0.52),0.2)]:
    ax0.add_patch(plt.Rectangle((r[0][0]-r[1],r[0][1]-r[1]),2*r[1],2*r[1],fill=False,ec='k',lw=1.2))
specs=[('coin extérieur (θ = 45°)',(1.414,1.414),0.22),('coin du trou (θ = 45°)',(0.707,0.707),0.22),('mur extérieur (θ ≈ 15°)',(1.93,0.52),0.2)]
for n,(t,c0,R) in enumerate(specs):
    ax=fig.add_axes([0.53+ (n%2)*0.235, 0.52 - (n//2)*0.46, 0.22, 0.40]); draw(ax,a,c0,R,lw=0.6); ax.set_title(t,fontsize=10)
fig.legend(handles=leg,loc='lower right',bbox_to_anchor=(0.98,0.08),fontsize=10)
fig.suptitle(title,fontsize=13)
plt.savefig(out,dpi=120)
