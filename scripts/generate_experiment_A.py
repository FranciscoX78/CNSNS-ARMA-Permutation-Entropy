#!/usr/bin/env python3
"""CNSNS synthetic experiment A: fixed stationary AR(2) regimes, m=6, N=5m!=3600.
Reproducibility: NumPy Generator seed 20261019, QMC Sobol seeds 4343 & 9917.
Generates independently per regime, avoids fitting mixed/nonstationary windows.
"""
import time, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal, linalg, special, stats
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent / 'data' / 'regenerated_A'
ROOT.mkdir(parents=True, exist_ok=True)
seed=20261019
rng=np.random.default_rng(seed)
m=6; N=5*math.factorial(m); TAU=1; BURN=1000; REPS=30; SURS=25
pole_grid=np.array([.95,.8,.6,.35,.0,-.35,-.6,-.8,-.95])
qmc_pow=16; oracle_pow=18
QMC_SEED=4343
start=time.time()


def pe_batch(x):
    """PE for one or several 1d windows, no ties expected in Gaussian series."""
    X=np.asarray(x,dtype=np.float64)
    if X.ndim == 1:X=X[None,:]
    assert X.shape[1] == N
    view=np.lib.stride_tricks.sliding_window_view(X,m,axis=1)
    # (n_series, N-m+1, m), tau=1
    pats=np.argsort(view,axis=-1)
    codes=np.zeros(pats.shape[:-1],dtype=np.int32)
    for j in range(m): codes=codes*m+pats[...,j]
    codes += (np.arange(len(X),dtype=np.int32)[:,None])*(m**m)
    counts=np.bincount(codes.ravel(),minlength=len(X)*(m**m)).reshape(len(X),m**m)
    P=counts / (N-m+1)
    h=-np.sum(special.xlogy(P,P),axis=1)/np.log(math.factorial(m))
    return h


def ent_pattern_samples(y):
    """Population pattern entropy approximation from independent Gaussian embedded vectors."""
    pat=np.argsort(y,axis=1)
    code=np.zeros(len(y),dtype=np.int32)
    for j in range(m):code=code*m+pat[:,j]
    counts=np.bincount(code,minlength=m**m)
    P=counts[counts>0]/len(y)
    return float(-np.sum(P*np.log(P))/np.log(math.factorial(m)))


def roots(phi1,phi2):
    return np.roots([1,-phi1,-phi2])


def H_gauss_qmc(phi1,phi2,z):
    """Gaussian ordinal entropy from stationary covariance (numerical orthant integration)."""
    pp=roots(phi1,phi2)
    if np.max(np.abs(pp)) >= 1:
        return float('nan')
    rhos=np.zeros(m);rhos[0]=1
    rhos[1]=phi1/(1-phi2)
    for k in range(2,m): rhos[k]=phi1*rhos[k-1]+phi2*rhos[k-2]
    C=linalg.toeplitz(rhos)
    chol=np.linalg.cholesky(C)
    y=z@chol.T
    return ent_pattern_samples(y)


def fit_ar2_many(X):
    """Conditional least-squares AR(2), intercept included, each realization separately."""
    y=X[:,2:]
    x1=X[:,1:-1]
    x2=X[:,:-2]
    n=y.shape[1]
    xx=np.zeros((len(X),3,3))
    xx[:,0,0]=n
    xx[:,0,1]=xx[:,1,0]=x1.sum(1)
    xx[:,0,2]=xx[:,2,0]=x2.sum(1)
    xx[:,1,1]=(x1*x1).sum(1)
    xx[:,1,2]=xx[:,2,1]=(x1*x2).sum(1)
    xx[:,2,2]=(x2*x2).sum(1)
    xy=np.stack([y.sum(1),(x1*y).sum(1),(x2*y).sum(1)],axis=1)
    pars=np.linalg.solve(xx,xy[...,None])[...,0]
    resid=y-pars[:,0,None]-pars[:,1,None]*x1-pars[:,2,None]*x2
    var=np.mean(resid**2,axis=1)
    return pars,var

z_fit=special.ndtri(qmc.Sobol(m,scramble=True,seed=QMC_SEED).random_base2(qmc_pow))
z_oracle=special.ndtri(qmc.Sobol(m,scramble=True,seed=9917).random_base2(oracle_pow))
# extra oracle replication for checking QMC numerical precision
z_oracle2=special.ndtri(qmc.Sobol(m,scramble=True,seed=77117).random_base2(oracle_pow))
rows=[]
sur_records=[]
for ki,r in enumerate(pole_grid):
    r2=.35*r
    phi1=r+r2; phi2=-r*r2
    if not (np.max(np.abs(roots(phi1,phi2))) < 1):raise ValueError('unstable model')
    innov=rng.normal(size=(REPS,N+BURN))
    data=signal.lfilter([1.],[1.,-phi1,-phi2],innov,axis=-1)[:,BURN:]
    empirical=pe_batch(data)
    params, var=fit_ar2_many(data)
    htrue=H_gauss_qmc(phi1,phi2,z_oracle)
    htrue2=H_gauss_qmc(phi1,phi2,z_oracle2)
    errs=np.abs(htrue-htrue2)
    print(f'Regime {ki+1}/{len(pole_grid)} dominant_true={r:+.2f} Htrue={htrue:.5f} QMC agreement={errs:.5g}',flush=True)
    for j in range(REPS):
        a,b,c=params[j]
        pr=roots(b,c)
        ip=np.argmax(np.abs(pr))
        dom=pr[ip]
        stable=bool(np.max(np.abs(pr))<1)
        hfit=H_gauss_qmc(b,c,z_fit) if stable else np.nan
        # matched-length Gaussian AR2 surrogates, fitted moments, gaussian innovation
        # Same 3600 length and identical PE estimation scheme.
        if stable:
            ee=rng.normal(scale=np.sqrt(var[j]),size=(SURS,N+BURN))
            s=signal.lfilter([1.],[1.,-b,-c],ee,axis=-1)[:,BURN:]
            hsur=pe_batch(s)
            hsur_mean=float(hsur.mean());hsur_std=float(hsur.std(ddof=1))
            hsur_lo,hsur_hi=np.quantile(hsur,[.025,.975])
            sur_records.extend(dict(regime=ki+1,window=ki*REPS+j+1,surrogate=si+1,PE_surrogate=float(hval)) for si,hval in enumerate(hsur))
        else:
            hsur_mean=hsur_std=hsur_lo=hsur_hi=np.nan
        rows.append(dict(regime=ki+1,window=ki*REPS+j+1,replicate=j+1,
                         pole1_true=r,pole2_true=r2,
                         phi1_true=phi1,phi2_true=phi2,
                         phi0_est=a,phi1_est=b,phi2_est=c,
                         innovation_variance_est=var[j],stationary_fit=stable,
                         pole_dom_est_real=float(np.real(dom)),
                         pole_dom_est_imag=float(np.imag(dom)),
                         pole_dom_est_abs=float(abs(dom)),
                         PE_emp=float(empirical[j]),
                         Hlin_oracle_gaussian=htrue,Hlin_oracle_gaussian_check=htrue2,
                         Hlin_fit_gaussian=hfit,
                         Hlin_fit_matched_N=hsur_mean,
                         Hlin_fit_matched_N_sd=hsur_std,
                         Hlin_fit_matched_N_q025=hsur_lo,
                         Hlin_fit_matched_N_q975=hsur_hi))
    print('  completed windows',len(rows),'elapsed',round(time.time()-start,1),'s',flush=True)

D=pd.DataFrame(rows)
D.to_csv(ROOT/'CNSNS_synthetic_A_all_windows.csv',index=False)
S=pd.DataFrame(sur_records)
S.to_csv(ROOT/'CNSNS_synthetic_A_surrogate_entropies.csv',index=False)

G=D.groupby('regime',sort=True).agg(pole1_true=('pole1_true','first'),pole2_true=('pole2_true','first'),
    phi1_true=('phi1_true','first'),phi2_true=('phi2_true','first'),
    n=('PE_emp','size'),valid_fits=('stationary_fit','sum'),
    PE_mean=('PE_emp','mean'),PE_sd=('PE_emp','std'),
    Hlin_oracle=('Hlin_oracle_gaussian','mean'),Hlin_fit_mean=('Hlin_fit_gaussian','mean'),
    Hlin_matched_N_mean=('Hlin_fit_matched_N','mean'),
    pole_dom_est_mean=('pole_dom_est_real','mean'),
    phi1_est_mean=('phi1_est','mean'),phi2_est_mean=('phi2_est','mean')).reset_index()
G.to_csv(ROOT/'CNSNS_synthetic_A_regime_summary.csv',index=False)

pairs=D[['PE_emp','Hlin_oracle_gaussian','Hlin_fit_gaussian','Hlin_fit_matched_N']].dropna()
metrics={
    'n_total_windows':len(D),
    'n_valid_AR2_fit':int(D.stationary_fit.sum()),
    'pe_hlin_fitted_pearson_window':float(pairs.PE_emp.corr(pairs.Hlin_fit_gaussian)),
    'pe_hlin_fitted_pearson_regime':float(G.PE_mean.corr(G.Hlin_fit_mean)),
    'pe_hlin_matchedN_pearson_window':float(pairs.PE_emp.corr(pairs.Hlin_fit_matched_N)),
    'pe_hlin_matchedN_pearson_regime':float(G.PE_mean.corr(G.Hlin_matched_N_mean)),
    'true_pole_fitted_pole_rmse':float(np.sqrt(np.mean((D.pole_dom_est_real-D.pole1_true)**2))),
    'true_phi1_fit_rmse':float(np.sqrt(np.mean((D.phi1_est-D.phi1_true)**2))),
    'true_phi2_fit_rmse':float(np.sqrt(np.mean((D.phi2_est-D.phi2_true)**2))),
    'PEminus_Hlin_fit_bias':float(np.mean(pairs.PE_emp-pairs.Hlin_fit_gaussian)),
    'PEminus_Hlin_fit_rmse':float(np.sqrt(np.mean((pairs.PE_emp-pairs.Hlin_fit_gaussian)**2))),
    'PEminus_Hlin_matchedN_bias':float(np.mean(pairs.PE_emp-pairs.Hlin_fit_matched_N)),
    'PEminus_Hlin_matchedN_rmse':float(np.sqrt(np.mean((pairs.PE_emp-pairs.Hlin_fit_matched_N)**2))),
    'oracle_qmc_max_diff':float(np.max(np.abs(G.Hlin_oracle.values-D.groupby('regime').Hlin_oracle_gaussian_check.first().values))),
    'model':'AR(2) Gaussian innovation N(0,1); r2=0.35 r1; piecewise stationary independent simulated blocks',
    'm':m,'tau':TAU,'N_per_window':N,'regime_windows':REPS,'number_regimes':len(pole_grid),
    'burnin':BURN,'n_matched_surrogates_per_fit':SURS,
    'gaussian_reference_method':'Scrambled Sobol QMC 2^16 for fitted, 2^18 for oracle; gaussian covariance of 6 consecutive observations',
}
metrics['pe_hlin_matchedN_coverage_95surrogate']=float(np.mean((D.PE_emp>=D.Hlin_fit_matched_N_q025)&(D.PE_emp<=D.Hlin_fit_matched_N_q975)))
# Symmetric mean absolute percentage error on normalized entropies, as a fraction.
for pred in ['Hlin_fit_gaussian','Hlin_fit_matched_N']:
    valid = D[['PE_emp', pred]].dropna()
    smape = np.mean(2*np.abs(valid['PE_emp']-valid[pred])/(np.abs(valid['PE_emp'])+np.abs(valid[pred])))
    metrics['symmetric_relative_error_'+pred]=float(smape)

# Jensen-Shannon divergence of the sampling distributions of PE in each regime.
# Fixed rule: 10 equally spaced bins over the union of all empirical and surrogate PE values.
# Null benchmark: independent draws of one surrogate per window, compared with surrogate pool.
jsrows=[]
js_rng=np.random.default_rng(912345)
for ki in range(1,len(pole_grid)+1):
    e=D.loc[D.regime==ki,'PE_emp'].to_numpy()
    so=S.loc[S.regime==ki,'PE_surrogate'].to_numpy()
    edges=np.linspace(min(e.min(),so.min())-1e-9,max(e.max(),so.max())+1e-9,11)
    def hist(v):
        c=np.histogram(v,bins=edges)[0].astype(float)
        return c/c.sum()
    obs=float(stats.entropy(hist(e),0.5*(hist(e)+hist(so)),base=2)/2 + stats.entropy(hist(so),0.5*(hist(e)+hist(so)),base=2)/2)
    srv=S.loc[S.regime==ki,['window','surrogate','PE_surrogate']].pivot_table(index='window',columns='surrogate',values='PE_surrogate').to_numpy()
    null=[]
    for q in range(500):
        sim=srv[np.arange(REPS),js_rng.integers(0,SURS,size=REPS)]
        h1,h2=hist(sim),hist(so)
        mix=.5*(h1+h2)
        null.append(float(.5*stats.entropy(h1,mix,base=2)+.5*stats.entropy(h2,mix,base=2)))
    jsrows.append(dict(regime=ki,pole_dom_true=float(pole_grid[ki-1]),JSD_bits=obs,
                       JSD_null_median=float(np.median(null)),
                       JSD_null_q025=float(np.quantile(null,.025)),
                       JSD_null_q975=float(np.quantile(null,.975)),
                       is_below_null_975=bool(obs<=np.quantile(null,.975))))
js=pd.DataFrame(jsrows)
js.to_csv(ROOT/'CNSNS_synthetic_A_JSD_per_regime.csv',index=False)
metrics['JS_regimes_within_null95']=int(js.is_below_null_975.sum())
metrics['JS_mean_bits']=float(js.JSD_bits.mean())
with open(ROOT/'CNSNS_synthetic_A_metrics.json','w') as f:json.dump(metrics,f,indent=2,ensure_ascii=False)

# graphics
fig,axs=plt.subplots(3,1,figsize=(13,10),sharex=True,layout='constrained')
xx=D.window.values
cols=['#174e87','#e07b39','#529c65','#8d58a9']
axs[0].scatter(xx,D.PE_emp,s=8,alpha=.35,c=cols[0],label='PE empírica (cada ventana)')
axs[0].plot(G.regime*REPS-(REPS-1)/2,G.PE_mean,marker='o',linewidth=2.0,color=cols[0],label='PE: media por régimen')
axs[0].plot(G.regime*REPS-(REPS-1)/2,G.Hlin_oracle,marker='s',linewidth=2.0,color=cols[1],label='Hlin teórica (AR(2) verdadero, QMC)')
axs[0].plot(G.regime*REPS-(REPS-1)/2,G.Hlin_fit_mean,marker='^',linewidth=2.0,color=cols[2],label='Hlin (AR(2) estimado, QMC)')
axs[0].plot(G.regime*REPS-(REPS-1)/2,G.Hlin_matched_N_mean,marker='d',linewidth=2.0,color=cols[3],label='Referencia AR(2) estimado, realizaciones N=3600')
axs[0].set_ylabel('PE normalizada (m=6)')
axs[0].legend(fontsize=8,ncol=2,loc='best')
axs[0].set_title('Experimento A — transición controlada de polos AR(2) gaussianos')
axs[1].scatter(xx,D.pole_dom_est_real,s=7,alpha=.35,c=cols[0],label='Polo dominante estimado (por ventana)')
for ridx,row in G.iterrows():
    axs[1].hlines(row.pole1_true, ridx*REPS+.5, (ridx+1)*REPS+.5, color=cols[1],lw=2,label='Polo dominante verdadero' if ridx==0 else None)
axs[1].plot(G.regime*REPS-(REPS-1)/2,G.pole_dom_est_mean,color=cols[2],lw=2,marker='o',label='Polo estimado: media por régimen')
axs[1].set_ylabel('Polo dominante real')
axs[1].legend(fontsize=8)
axs[2].scatter(xx,D.PE_emp-D.Hlin_fit_gaussian,s=7,alpha=.45,c=cols[1],label='PE − Hlin teórica (modelo estimado)')
axs[2].scatter(xx,D.PE_emp-D.Hlin_fit_matched_N,s=7,alpha=.35,c=cols[2],label='PE − referencia de igual longitud')
axs[2].axhline(0,color='black',lw=0.8)
axs[2].set_ylabel('Diferencia de entropías')
axs[2].set_xlabel('Índice de ventana sintética (30 ventanas por régimen)')
axs[2].legend(fontsize=8)
for ax in axs:
    for i in range(1,len(G)):ax.axvline(i*REPS+.5,lw=.75,color='gray',alpha=.4)
    ax.grid(alpha=.2)
for row in [axs[0],axs[1],axs[2]]:
    sec=row.secondary_xaxis('top' if row is axs[0] else 'bottom') if False else None
fig.savefig(ROOT/'CNSNS_synthetic_A_main.png',dpi=175)
plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
axs[0].scatter(D.Hlin_fit_gaussian,D.PE_emp,s=16,alpha=.55)
lims=[min(D.Hlin_fit_gaussian.min(),D.PE_emp.min())-.015,max(D.Hlin_fit_gaussian.max(),D.PE_emp.max())+.015]
axs[0].plot(lims,lims,'k--',alpha=.5)
axs[0].set(xlabel='Hlin gaussian estimated AR(2)',ylabel='Empirical PE',title='Población vs estimador finito')
axs[1].scatter(D.Hlin_fit_matched_N,D.PE_emp,s=16,alpha=.55)
lims=[min(D.Hlin_fit_matched_N.min(),D.PE_emp.min())-.015,max(D.Hlin_fit_matched_N.max(),D.PE_emp.max())+.015]
axs[1].plot(lims,lims,'k--',alpha=.5)
axs[1].set(xlabel='Reference PE, same length N=3600',ylabel='Empirical PE',title='PE empírica vs surrogados de longitud igual')
for ax in axs:ax.grid(alpha=.2)
fig.savefig(ROOT/'CNSNS_synthetic_A_scatter.png',dpi=175)
plt.close(fig)

print('METRICS:',json.dumps(metrics,indent=2,ensure_ascii=False),flush=True)
print('GROUP SUMMARY\n',G.to_string(index=False),flush=True)
print('ELAPSED:',round(time.time()-start,1),flush=True)
