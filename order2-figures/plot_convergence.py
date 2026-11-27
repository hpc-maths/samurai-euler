# Trace convergence_rotation.csv. Usage : python plot_convergence.py
import csv, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rows=[r for r in csv.reader(l for l in open('convergence_rotation.csv') if not l.startswith('#'))][1:]
data={}
for c,L,*v in rows: data.setdefault(c,[]).append([int(L)]+[float(x) for x in v])
style={'ordre1':('ordre 1','#7f7f7f','o-'),'ordre2_none':('ordre 2, sans limiteur','#1f77b4','s-'),'ordre2_moncen':('ordre 2, moncen','#2ca02c','^-'),'ordre2_none_sans_courbure':('ordre 2, sans correction de courbure','#d62728','v--')}
fig,axs=plt.subplots(1,3,figsize=(16,5))
for ax,(col,title) in zip(axs,[(1,'L¹ |p − p_ex| (tout le domaine)'),(2,'L¹ |p − p_ex| près des murs'),(4,'L¹ |v − v_ex|')]):
    for c,(lab,color,st) in style.items():
        a=np.array(data[c]); h=2.0**-a[:,0]; e=a[:,col]
        ax.loglog(h,e,st,color=color,label=f'{lab} (pente {np.log2(e[-2]/e[-1]):.2f})')
    h=np.array([2**-4,2**-6]); e0=data['ordre2_none'][0][col]
    ax.loglog(h,e0*(h/h[0])**2,'k:',lw=1,label='pente 2'); ax.loglog(h,data['ordre1'][0][col]*(h/h[0]),'k-.',lw=1,label='pente 1')
    ax.set_xlabel('h (pas paramétrique)'); ax.set_title(title); ax.grid(True,which='both',alpha=0.3); ax.legend(fontsize=8)
fig.suptitle('Rotation rigide stationnaire sur la carte optimisée, t = 1 (niveaux 4, 5, 6)')
plt.tight_layout(); plt.savefig('convergence_rotation.png',dpi=110)
