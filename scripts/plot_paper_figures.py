#!/usr/bin/env python3
"""Re-create approved paper figure panel values from committed record/window CSVs.
The figures are analytical reproductions, not guaranteed pixel-identical to earlier artwork.
"""
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures'/'regenerated';FIG.mkdir(parents=True,exist_ok=True)
N=3600

def make_A():
    d=pd.read_csv(ROOT/'data/experiment_A/CNSNS_synthetic_A_all_windows.csv')
    pgrid=[.95,.8,.6,.35,0,-.35,-.6,-.8,-.95]
    rng=np.random.default_rng(20261019)
    blocks=[]
    for r in pgrid:
        r2=.35*r
        innov=rng.normal(size=(30,N+1000))
        data=signal.lfilter([1.],[1.,-(r+r2),r*r2],innov,axis=-1)[:,1000:]
        blocks.append(data[0])
    d['symrel']=100*2*np.abs(d.PE_emp-d.Hlin_fit_gaussian)/(np.abs(d.PE_emp)+np.abs(d.Hlin_fit_gaussian))
    g=d.groupby('regime').agg(PE=('PE_emp','mean'),PEsd=('PE_emp','std'),H=('Hlin_fit_gaussian','mean'),Hsd=('Hlin_fit_gaussian','std'),
         root=('pole_dom_est_real','mean'),rootsd=('pole_dom_est_real','std'),truth=('pole1_true','first'),err=('symrel','mean'),errsd=('symrel','std')).reset_index()
    cent=(np.arange(9)+.5)*N;fig,ax=plt.subplots(4,1,figsize=(12,9.7),sharex=False,layout='constrained')
    ax[0].plot(np.concatenate(blocks),linewidth=.65);ax[0].set_ylabel(r'$x_t$')
    ax[1].errorbar(cent,g.PE,yerr=g.PEsd,fmt='o-',capsize=3,label='Empirical PE')
    ax[1].errorbar(cent,g.H,yerr=g.Hsd,fmt='s-',capsize=3,label=r'$H_{lin}$');ax[1].legend()
    ax[1].set_ylabel('Normalized entropy')
    ax[2].errorbar(cent,g.root,yerr=g.rootsd,fmt='o--',capsize=3,label='Estimated dominant root')
    ax[2].plot(cent,g.truth,'s-',label='True dominant root');ax[2].legend();ax[2].set_ylabel('Signed reciprocal root')
    ax[3].errorbar(cent,g.err,yerr=g.errsd,fmt='o-',capsize=3);ax[3].set_ylabel('Symmetric relative error (%)')
    for i,a in enumerate(ax):
        for b in np.arange(1,9)*N:a.axvline(b,color='gray',ls=':',lw=.8)
        a.set_xlim(0,9*N);a.grid(alpha=.16)
        a.text(.01,.9,'ABCD'[i],transform=a.transAxes,fontweight='bold')
    ax[3].set_xticks(cent,[str(i) for i in range(1,10)]);ax[3].set_xlabel('Regime (9 independent Gaussian AR(2) blocks)')
    fig.savefig(FIG/'experiment_A.png',dpi=160);plt.close(fig)

def make_B():
    d=pd.read_csv(ROOT/'data/experiment_B/CNSNS_synthetic_B_all_windows_AR1.csv')
    # original innovation generation independent of reference simulations
    rng=np.random.default_rng(20261020)
    vals=[]
    for a in [0,.2,.4,.6,.8,1,1.25,1.5,2]:
        innov=rng.normal(size=(30,N+1000))
        x=np.zeros_like(innov)
        for t in range(1,N+1000):x[:,t]=.55*x[:,t-1]+a*np.sin(x[:,t-1])+innov[:,t]
        vals.append(x[0,1000:])
    d['delta']=d.PE_emp-d.AR1_ref_mean
    d['symrel']=100*2*np.abs(d.delta)/(np.abs(d.PE_emp)+np.abs(d.AR1_ref_mean))
    g=d.groupby('regime').agg(PE=('PE_emp','mean'),PEsd=('PE_emp','std'),H=('Hlin_AR1_population','mean'),Hsd=('Hlin_AR1_population','std'),
         gap=('delta','mean'),gapsd=('delta','std'),err=('symrel','mean'),errsd=('symrel','std')).reset_index()
    cent=(np.arange(9)+.5)*N;fig,ax=plt.subplots(4,1,figsize=(12,9.7),sharex=False,layout='constrained')
    ax[0].plot(np.concatenate(vals),lw=.5);ax[0].set_ylabel(r'$x_t$')
    ax[1].errorbar(cent,g.PE,yerr=g.PEsd,fmt='o-',capsize=3,label='Empirical PE')
    ax[1].errorbar(cent,g.H,yerr=g.Hsd,fmt='s-',capsize=3,label=r'$H_{lin}$, fitted AR(1)')
    ax[1].legend();ax[1].set_ylabel('Normalized entropy')
    ax[2].errorbar(cent,g.gap,yerr=g.gapsd,fmt='o-',capsize=3);ax[2].axhline(0,color='gray',lw=.9)
    ax[2].set_ylabel(r'$\Delta H$, matched $N$')
    ax[3].errorbar(cent,g.err,yerr=g.errsd,fmt='o-',capsize=3);ax[3].set_ylabel('Symmetric relative error (%)')
    for i,a in enumerate(ax):
        for b in np.arange(1,9)*N:a.axvline(b,color='gray',ls=':',lw=.8)
        a.set_xlim(0,9*N);a.grid(alpha=.16);a.text(.01,.9,'ABCD'[i],transform=a.transAxes,fontweight='bold')
    ax[3].set_xticks(cent,['0','.2','.4','.6','.8','1','1.25','1.5','2']);ax[3].set_xlabel(r'Nonlinear perturbation $\alpha$')
    fig.savefig(FIG/'experiment_B.png',dpi=160);plt.close(fig)

def make_ims():
    ds={b:pd.read_csv(ROOT/f'data/ims/record_level_m4/IMS_Set2_B{b}_m4_record_summaries_982.csv') for b in range(1,5)}
    def plot3(axs,df):
        x=df.record_idx
        for col,j in [('PE',0),('Hlin',0),('pole_mod',1),('symrelerr',2)]:
            axs[j].plot(x,df[col],alpha=.18,lw=.7)
            axs[j].plot(x,df[col+'_smooth'],lw=1.6,label=('Empirical PE' if col=='PE' else r'$H_{lin}$' if col=='Hlin' else None))
        axs[0].set_ylabel('Entropy');axs[0].legend(fontsize=9)
        axs[1].set_ylabel('Dominant root modulus');axs[2].set_ylabel('Symmetric relative error (%)')
        for j,ax in enumerate(axs):ax.grid(alpha=.2);ax.text(.01,.9,'ABC'[j],transform=ax.transAxes,fontweight='bold')
        axs[2].set_xlabel('Record index')
    fig,ax=plt.subplots(3,1,figsize=(8.5,8.5),sharex=True,layout='constrained');plot3(ax,ds[1]);fig.savefig(FIG/'ims_main_bearing1.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(4,3,figsize=(12,11),sharex='col',layout='constrained')
    for b in range(1,5):
        df=ds[b];x=df.record_idx
        for col,j in [('PE',0),('Hlin',0),('pole_mod',1),('symrelerr',2)]:
            ax[b-1,j].plot(x,df[col],alpha=.13,lw=.65)
            ax[b-1,j].plot(x,df[col+'_smooth'],lw=1.4)
        ax[b-1,0].set_ylabel(f'Bearing {b}\nEntropy');ax[b-1,1].set_ylabel('Root modulus');ax[b-1,2].set_ylabel('Error (%)')
        for a in ax[b-1]:a.grid(alpha=.2)
    for a in ax[-1]:a.set_xlabel('Record index')
    for j,t in enumerate(['PE and Hlin','Dominant root modulus','Symmetric relative error']):ax[0,j].set_title(t)
    fig.savefig(FIG/'ims_appendix_all4.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['all','A','B','ims'],default='all');a=p.parse_args()
    if a.part in ('all','A'):make_A()
    if a.part in ('all','B'):make_B()
    if a.part in ('all','ims'):make_ims()
    print('Saved figures to',FIG)
