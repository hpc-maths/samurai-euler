# rho, p - p_equilibre, v_r, v_theta - v_theta_equilibre a t = 0, pi/4, pi/2 pour un mur.
# Usage : python plot_uniform_detail.py . normal sortie.png
import sys, numpy as np
sys.path.insert(0, sys.argv[1]); from vtu import read_vtu
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
w=sys.argv[2]; out=sys.argv[3]
def pex(r):
    o=0.1; return np.where(r<1.5, 1+0.5*o*o*1*(r*r-1), 1+0.5*o*o*(1*(1.5**2-1)+2*(r*r-1.5**2)))
def vth(r): return np.where(r<1.5,-0.1*r,0.1*r)
frames=[('0000','t = 0'),('0002','t = π/4'),('0004','t = π/2')]
fig,axs=plt.subplots(3,4,figsize=(22,16))
for i,(k,tl) in enumerate(frames):
    a=read_vtu(f'kh_uniform7_{w}_{k}.vtu'); c=np.array([p.mean(0) for p in a['polys']]); r=np.hypot(c[:,0],c[:,1])
    vr=(c[:,0]*a['u']+c[:,1]*a['v'])/r; vt=(-c[:,1]*a['u']+c[:,0]*a['v'])/r
    fields=[(a['rho'],'densité ρ','viridis',(0.95,2.05)),(a['p']-pex(r),'p − p_équilibre(r)','RdBu_r',None),(vr,'vitesse radiale v_r','RdBu_r',None),(vt-vth(r),'v_θ − v_θ,équilibre(r)','RdBu_r',None)]
    for j,(v,lab,cmap,cl) in enumerate(fields):
        ax=axs[i,j]; pc=PolyCollection(a['polys'],array=v,cmap=cmap,edgecolors='face',linewidths=0.05)
        if cl is None:
            m=np.percentile(np.abs(v),99.9); m=m if m>0 else 1e-12; pc.set_clim(-m,m)
        else: pc.set_clim(*cl)
        ax.add_collection(pc); ax.set_xlim(-2.05,2.05); ax.set_ylim(-2.05,2.05); ax.set_aspect('equal')
        ax.set_title(f'{lab}, {tl}\nmax |·| = {np.abs(v).max():.2e}' if cl is None else f'{lab}, {tl}',fontsize=11); fig.colorbar(pc,ax=ax,fraction=0.046)
fig.suptitle(f"Kelvin–Helmholtz cylindrique, mur {w}, carte optimisée, grille uniforme niveau 7 (196 608 cellules). Échelles propres à chaque panneau (99,9e centile).",fontsize=14)
plt.tight_layout(rect=[0,0,1,0.965]); plt.savefig(out,dpi=90)
