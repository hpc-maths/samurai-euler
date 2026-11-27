# Comparaison mur normal / mur cartesien (rho, p, v_r) a un instant.
# Usage : python plot_uniform_compare.py . 0004 sortie.png "t = pi/2"
import sys, numpy as np
sys.path.insert(0, sys.argv[1]); from vtu import read_vtu
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
rows=[('normal','mur normal (réflexion selon la normale physique)'),('cartesian','mur cartésien (ancien Reflective)')]
k=sys.argv[2]; out=sys.argv[3]; tlabel=sys.argv[4]
fig,axs=plt.subplots(2,3,figsize=(18,11.5))
lims={}
data={}
for w,_ in rows:
    a=read_vtu(f'kh_uniform7_{w}_{k}.vtu'); c=np.array([p.mean(0) for p in a['polys']]); r=np.hypot(c[:,0],c[:,1])
    vr=(c[:,0]*a['u']+c[:,1]*a['v'])/r
    data[w]=(a,{'rho':a['rho'],'p':a['p'],'vr':vr})
for i,(w,title) in enumerate(rows):
    a,f=data[w]
    for j,(key,lab,cmap) in enumerate([('rho','densité ρ','viridis'),('p','pression p','magma'),('vr','vitesse radiale v_r','RdBu_r')]):
        ax=axs[i,j]; v=f[key]
        pc=PolyCollection(a['polys'],array=v,cmap=cmap,edgecolors='face',linewidths=0.05)
        if key=='vr':
            m=max(np.abs(data['normal'][1]['vr']).max(),1e-12); m=max(m,np.abs(data['cartesian'][1]['vr']).max()); pc.set_clim(-m,m)
        else:
            lo=min(data['normal'][1][key].min(),data['cartesian'][1][key].min()); hi=max(data['normal'][1][key].max(),data['cartesian'][1][key].max()); pc.set_clim(lo,hi)
        ax.add_collection(pc); ax.set_xlim(-2.05,2.05); ax.set_ylim(-2.05,2.05); ax.set_aspect('equal')
        ax.set_title(f'{lab}\n{title}',fontsize=10); fig.colorbar(pc,ax=ax,fraction=0.046)
fig.suptitle(f'Kelvin–Helmholtz cylindrique, carte optimisée, grille uniforme niveau 7 (512 × 512 paramétrique, 196 608 cellules), {tlabel}',fontsize=13)
plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(out,dpi=100)
