#!/usr/bin/env python3
"""Independent AR(1)-only replication of Experiment B.

For the exact numerical values of the submitted manuscript, use frozen CSV files
in data/experiment_B. This AR(1)-only script uses the original signal seed but
an AR(1)-only Monte Carlo stream; surrogate means may differ slightly from the
original recorded execution, which interleaved unused auxiliary computations.
"""
import sys, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import special, linalg, signal
from scipy.stats import qmc
REPO=Path(__file__).resolve().parent.parent
OUT=REPO/'data'/'regenerated_B_AR1';OUT.mkdir(parents=True,exist_ok=True)
M=6;N=5*math.factorial(M);BURN=1000;REPS=30;SURS=100
ALPHAS=[0,0.2,0.4,0.6,0.8,1,1.25,1.5,2]
rng=np.random.default_rng(20261020)
srng=np.random.default_rng(20261020+151)
Z=special.ndtri(qmc.Sobol(d=M,scramble=True,seed=5702).random_base2(16))

def pe(X):
    X=np.asarray(X)
    if X.ndim==1:X=X[None,:]
    P=np.argsort(np.lib.stride_tricks.sliding_window_view(X,M,axis=1),axis=-1,kind='stable')
    code=np.zeros(P.shape[:2],dtype=np.int32)
    for j in range(M):code=code*M+P[:,:,j]
    H=[]
    for i in range(len(X)):
        cnt=np.bincount(code[i],minlength=M**M)
        prob=cnt[cnt>0]/code.shape[1]
        H.append(-np.sum(prob*np.log(prob))/np.log(math.factorial(M)))
    return np.array(H)

def hlin(phi):
    cov=linalg.toeplitz(phi**np.arange(M))
    ords=np.argsort(Z@np.linalg.cholesky(cov).T,axis=1)
    codes=np.zeros(len(Z),dtype=np.int32)
    for j in range(M): codes=codes*M+ords[:,j]
    p=np.bincount(codes,minlength=M**M)
    p=p[p>0]/len(Z)
    return -np.sum(p*np.log(p))/np.log(math.factorial(M))

def fit_ar1(X):
    means=X.mean(1)
    left=X[:,:-1]-means[:,None]
    right=X[:,1:]-means[:,None]
    phi=np.sum(left*right,axis=1)/np.sum(left*left,axis=1)
    intercept=X[:,1:].mean(1)-phi*X[:,:-1].mean(1)
    resid=X[:,1:]-intercept[:,None]-phi[:,None]*X[:,:-1]
    return phi,intercept,np.mean(resid**2,axis=1)

rows=[]
for i,alpha in enumerate(ALPHAS):
    innov=rng.normal(size=(REPS,N+BURN))
    X=np.zeros_like(innov)
    for j in range(1,N+BURN):
        X[:,j]=0.55*X[:,j-1]+alpha*np.sin(X[:,j-1])+innov[:,j]
    X=X[:,BURN:]
    Hp=pe(X);phis,intercepts,var=fit_ar1(X)
    for j in range(REPS):
        if abs(phis[j])>=1:raise ValueError('Nonstationary fitted AR(1) in output; do not silently drop')
        eps=srng.normal(scale=np.sqrt(var[j]),size=(SURS,N+BURN))
        ss=signal.lfilter([1.],[1.,-phis[j]],eps,axis=1)[:,BURN:]
        hs=pe(ss)
        rows.append(dict(alpha=alpha,regime=i+1,replicate=j+1,PE_emp=float(Hp[j]),
                         AR1_phi=float(phis[j]),Hlin_AR1_population=float(hlin(phis[j])),
                         AR1_ref_mean=float(hs.mean()),AR1_ref_sd=float(hs.std(ddof=1)),
                         AR1_ref_q025=float(np.quantile(hs,0.025)),AR1_ref_q975=float(np.quantile(hs,0.975))))
    print('Completed alpha =',alpha,flush=True)

D=pd.DataFrame(rows)
D.to_csv(OUT/'B_AR1_only_all_windows.csv',index=False)
print('Saved',OUT/'B_AR1_only_all_windows.csv')
