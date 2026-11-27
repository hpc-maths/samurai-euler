# Niveaux de raffinement de plusieurs sorties VTU.
# Usage : python plot_levels.py . "a.vtu|b.vtu" "titre a|titre b" sortie.png
import sys, numpy as np
sys.path.insert(0, sys.argv[1]); from vtu import read_vtu
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
files=sys.argv[2].split('|'); titles=sys.argv[3].split('|'); out=sys.argv[4]; field=sys.argv[5] if len(sys.argv)>5 else 'level'
nc=3; nr=(len(files)+nc-1)//nc; fig,axs=plt.subplots(nr,nc,figsize=(6.2*nc,6.0*nr),squeeze=False); axs=[axs.ravel()]
for ax,f,t in zip(axs[0],files,titles):
    a=read_vtu(f); v=a[field]
    pc=PolyCollection(a['polys'],array=v,cmap='viridis',edgecolors='k' if field=='level' else 'face',linewidths=0.08)
    if field=='level': pc.set_clim(3,6)
    ax.add_collection(pc); ax.set_xlim(-2.05,2.05); ax.set_ylim(-2.05,2.05); ax.set_aspect('equal'); ax.set_title(t,fontsize=10)
    fig.colorbar(pc,ax=ax,fraction=0.046,label=field)
plt.tight_layout(); plt.savefig(out,dpi=110)
