"""NASA IMS Set2: per-window empirical PE m=4,5,6 and low-order ARMA with Hlin.
N=5*m!, tau=1, nonoverlap in each original 20480-sample recording.
CSS constrained coordinate descent, approx BIC, same six model candidates as m=3.
m=4: exact Gaussian pattern probabilities using trivariate normal orthant formula.
m=5,6: numerical Gaussian reference by fixed Sobol quasi-Monte Carlo integration,
with plug-in entropy bias adjustment; not advertised as an exact analytic result.
"""
import os,sys,time,math,itertools
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import chi2,qmc,norm
from numba import njit,prange,set_num_threads
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=os.environ.get('IMS_WORK_DIR', str(Path(__file__).resolve().parent.parent / 'data' / 'intermediate'))
os.makedirs(ROOT,exist_ok=True)
CANDIDATES=np.array([[0,0],[1,0],[2,0],[0,1],[0,2],[1,1]],np.int32)

@njit(cache=True)
def legal_model(p,q,a,b):
    if p==1 and q==0:return -.995<a<.995
    if p==2:return 1.-a-b>.005 and 1.+a-b>.005 and 1.+b>.005 and 1.-b>.005
    if p==1 and q==1:return -.995<a<.995 and -.995<b<.995
    if q==1:return -.995<a<.995
    if q==2:return 1.+a+b>.005 and 1.-a+b>.005 and 1.+b>.005 and 1.-b>.005
    return True

@njit(cache=True)
def eval_css(z,p,q,a,b):
    if not legal_model(p,q,a,b):return 1e40,0.
    ar1=a if p>0 else 0.
    ar2=b if p==2 else 0.
    ma1=b if p==1 and q==1 else (a if p==0 and q>0 else 0.)
    ma2=b if p==0 and q==2 else 0.
    n=len(z);av1=av2=bv1=bv2=0.;saa=sab=sbb=0.
    # streaming CSS for innovation = a[t] - mu*b[t]. pre-sample innovations zero
    for t in range(n):
        aa=z[t] - ar1*(z[t-1] if t>0 else 0.) - ar2*(z[t-2] if t>1 else 0.)-ma1*av1-ma2*av2
        bb=1. - (ar1 if t>0 else 0.) - (ar2 if t>1 else 0.) -ma1*bv1-ma2*bv2
        av2=av1;av1=aa;bv2=bv1;bv1=bb
        saa+=aa*aa;sab+=aa*bb;sbb+=bb*bb
    mu=sab/sbb if sbb>1e-12 else 0.
    rss=max(0.,saa-sab*sab/sbb) if sbb>1e-12 else saa
    return rss,mu

@njit(cache=True)
def initialize(z):
    n=len(z);g0=g1=g2=0.
    for i in range(n):g0+=z[i]*z[i]
    for i in range(1,n):g1+=z[i]*z[i-1]
    for i in range(2,n):g2+=z[i]*z[i-2]
    g0/=n;g1/=n;g2/=n
    r1=g1/g0 if g0>1e-15 else 0.
    r2=g2/g0 if g0>1e-15 else 0.
    a1=max(-.9,min(.9,r1))
    a2=max(-.7,min(.7,(r2-r1*r1)/max(.05,1-r1*r1)))
    a21=max(-1.5,min(1.5,r1*(1-a2)))
    r=max(-.49,min(.49,r1))
    ma1=(2*r)/(1+np.sqrt(max(1e-12,1-4*r*r))) if abs(r)>1e-10 else 0.
    ma21=max(-.8,min(.8,r1));ma22=max(-.6,min(.6,r2))
    while not legal_model(0,2,ma21,ma22):ma21*=.7;ma22*=.7
    return a1,a21,a2,ma1,ma21,ma22

@njit(cache=True)
def optimize(z,p,q,a,b):
    v,mu=eval_css(z,p,q,a,b)
    if v>1e30:a=0.;b=0.;v,mu=eval_css(z,p,q,0.,0.)
    ndim=p+q
    # keep original m=3 constrained search (slightly fewer passes for long windows)
    for step in (0.40,0.18,0.07,0.025,0.008):
        for pas in range(2):
            for j in range(ndim):
                for sign in (-1,1):
                    aa=a+sign*step if j==0 else a
                    bb=b+sign*step if j==1 else b
                    vv,mm=eval_css(z,p,q,aa,bb)
                    if vv+1e-11<v:v=vv;mu=mm;a=aa;b=bb
    return v,mu,a,b

@njit(cache=True)
def empirical_pe(z,m,lookup):
    n=len(z)
    count=np.zeros(720,np.int32)
    ranks=np.zeros(6,np.int32)
    valid=ties=0
    for t in range(n-m+1):
        for i in range(m):ranks[i]=0
        tied=False
        for i in range(m-1):
            zi=z[t+i]
            for j in range(i+1,m):
                zj=z[t+j]
                if zi<zj:ranks[j]+=1
                elif zi>zj:ranks[i]+=1
                else:tied=True
        if tied:ties+=1;continue
        code=0
        for i in range(m):code+=ranks[i]<<(3*i)
        k=lookup[code]
        if k>=0:
            count[k]+=1;valid+=1
    if valid==0:return np.nan,valid,ties,0
    fact=1
    for ii in range(2,m+1):fact*=ii
    h=0.;distinct=0
    for i in range(fact):
        if count[i]>0:
            prob=count[i]/valid
            h-=prob*np.log(prob)
            distinct+=1
    return h/np.log(fact),valid,ties,distinct

@njit(cache=True)
def fit_window(z0,m,lookup):
    # output 20 cols: PE, valid, ties, distinct, rms, p, q, mu, innovvar, phi1,phi2,theta1,theta2,bic,css,LbQ5,LbQ10,status, ...
    o=np.empty(18,np.float64)
    for i in range(18):o[i]=np.nan
    pe,valid,ties,distinct=empirical_pe(z0,m,lookup)
    o[0]=pe;o[1]=valid;o[2]=ties;o[3]=distinct
    n=len(z0)
    mn=0.
    for v in z0:mn+=v
    mn/=n
    vv=0.
    for v in z0:vv+=(v-mn)**2
    vv/=n
    o[4]=np.sqrt(vv+mn*mn)
    if vv<1e-18:o[17]=1;return o
    sd=np.sqrt(vv);z=(z0-mn)/sd
    a1,a21,a2,ma1,ma21,ma22=initialize(z)
    pbest=0;qbest=0;bicbest=1e100;rssbest=1e100;mubest=0.;abest=0.;bbest=0.
    for mod in range(6):
        p=CANDIDATES[mod,0];q=CANDIDATES[mod,1]
        if mod==0:a=b=0.
        elif mod==1:a=a1;b=0.
        elif mod==2:a=a21;b=a2
        elif mod==3:a=ma1;b=0.
        elif mod==4:a=ma21;b=ma22
        else:a=a1;b=0.
        rss,mu,a,b=optimize(z,p,q,a,b)
        if mod==5:
            rss2,mu2,a02,b02=optimize(z,p,q,0.,ma1)
            if rss2<rss:rss,mu,a,b=rss2,mu2,a02,b02
        bic=n*np.log(max(rss/n,1e-20))+(p+q+2)*np.log(n)
        if bic<bicbest:pbest=p;qbest=q;bicbest=bic;rssbest=rss;mubest=mu;abest=a;bbest=b
    o[5]=pbest;o[6]=qbest;o[7]=mn+mubest*sd;o[8]=rssbest/n*vv
    o[9]=abest if pbest else 0.
    o[10]=bbest if pbest==2 else 0.
    o[11]=bbest if pbest==1 and qbest==1 else (abest if pbest==0 and qbest else 0.)
    o[12]=bbest if qbest==2 else 0.
    o[13]=bicbest;o[14]=rssbest/n*vv
    ar1=o[9];ar2=o[10];ma1=o[11];ma2=o[12]
    e=np.zeros(n,np.float64)
    for t in range(n):
        v=z[t]-mubest
        if t>0:v-=ar1*(z[t-1]-mubest)+ma1*e[t-1]
        if t>1:v-=ar2*(z[t-2]-mubest)+ma2*e[t-2]
        e[t]=v
    # LB Q, conditional residuals 5 and 10
    res_mean=np.mean(e)
    den=0.
    for t in range(n):den+=(e[t]-res_mean)**2
    if den>1e-18:
        qq=0.
        for k in range(1,11):
            numer=0.
            for t in range(k,n):numer+=(e[t]-res_mean)*(e[t-k]-res_mean)
            rr=numer/den
            qq+=rr*rr/(n-k)
            if k==5:o[15]=n*(n+2)*qq
        o[16]=n*(n+2)*qq
    o[17]=0
    return o

@njit(parallel=True,cache=True)
def run_fit(X,m,lookup):
    out=np.empty((len(X),18),np.float64)
    for j in prange(len(X)):
        out[j]=fit_window(X[j],m,lookup)
    return out

@njit(cache=True)
def acf_lowarma(p,q,phi1,phi2,theta1,theta2,m):
    out=np.empty(6,np.float64)
    for k in range(6):out[k]=0.
    if p==0 and q==0:
        out[0]=1.
    elif p==1 and q==0:
        for k in range(m):out[k]=phi1**k
    elif p==2:
        out[0]=1.
        out[1]=phi1/(1-phi2)
        for k in range(2,m):out[k]=phi1*out[k-1]+phi2*out[k-2]
    elif p==0 and q==1:
        out[0]=1.;out[1]=theta1/(1+theta1*theta1)
    elif p==0 and q==2:
        g0=1+theta1*theta1+theta2*theta2
        out[0]=1.;out[1]=theta1*(1+theta2)/g0;out[2]=theta2/g0
    else:
        g0=(1+theta1*theta1+2*phi1*theta1)/(1-phi1*phi1)
        out[0]=1.;out[1]=phi1+theta1/g0
        for k in range(2,m):out[k]=phi1*out[k-1]
    return out

@njit(cache=True)
def hlin_m4(ac,perms):
    probs=np.empty(24,np.float64)
    for idx in range(24):
        p=perms[idx]
        c=np.empty(3,np.float64)
        # every adjacent permuted difference var
        for i in range(3):
            a=p[i];b=p[i+1]
            c[i]=2*(1-ac[abs(b-a)])
        rsum=0.
        for i in range(3):
            for j in range(i+1,3):
                a=p[i];b=p[i+1];d=p[j];e=p[j+1]
                cov=ac[abs(b-e)]-ac[abs(b-d)]-ac[abs(a-e)]+ac[abs(a-d)]
                corr=max(-1.,min(1.,cov/np.sqrt(c[i]*c[j])))
                rsum+=np.arcsin(corr)
        probs[idx]=.125+rsum/(4*np.pi)
    probs=np.maximum(probs,0)
    probs/=np.sum(probs)
    h=0.
    for p in probs:
        if p>0:h-=p*np.log(p)
    return h/np.log(24)

@njit(parallel=True,cache=True)
def all_hlin4(fitted, perms):
    out=np.empty(len(fitted),np.float64)
    for i in prange(len(fitted)):
        if fitted[i,17]!=0:out[i]=np.nan;continue
        p=int(fitted[i,5]);q=int(fitted[i,6])
        ac=acf_lowarma(p,q,fitted[i,9],fitted[i,10],fitted[i,11],fitted[i,12],4)
        out[i]=hlin_m4(ac,perms)
    return out

@njit(cache=True)
def qmc_entropy(ac,m,u,lookup):
    # cholesky of theoretical Toeplitz covariance correlation matrix
    L=np.zeros((6,6),np.float64)
    for i in range(m):
        for j in range(i+1):
            s=ac[abs(i-j)]
            for k in range(j):s-=L[i,k]*L[j,k]
            if i==j:
                if s<=1e-12:return np.nan
                L[i,j]=np.sqrt(s)
            else:L[i,j]=s/L[j,j]
    counts=np.zeros(720,np.int32)
    x=np.empty(6,np.float64)
    ranks=np.empty(6,np.int32)
    for t in range(u.shape[0]):
        for i in range(m):
            v=0.
            for j in range(i+1):v+=L[i,j]*u[t,j]
            x[i]=v;ranks[i]=0
        for i in range(m-1):
            zi=x[i]
            for j in range(i+1,m):
                if zi<x[j]:ranks[j]+=1
                else:ranks[i]+=1
        code=0
        for i in range(m):code+=ranks[i]<<(3*i)
        idx=lookup[code]
        if idx>=0:counts[idx]+=1
    entropy=0.; k_nonzero=0
    total=u.shape[0]
    fact=1
    for ii in range(2,m+1):fact*=ii
    for i in range(fact):
        c=counts[i]
        if c>0:
            prob=c/total
            entropy-=prob*np.log(prob)
            k_nonzero+=1
    # Miller-Madow adjustment for entropy evaluated with a finite QMC sample;
    # it is an estimate of the theoretical model PE, not exact.
    entropy+=(k_nonzero-1)/(2*total)
    return min(1.,entropy/np.log(fact))

@njit(parallel=True,cache=True)
def all_hlin_qmc(fitted,m,lookup,u):
    out=np.empty(len(fitted),np.float64)
    for i in prange(len(fitted)):
        if fitted[i,17]!=0:out[i]=np.nan;continue
        p=int(fitted[i,5]);q=int(fitted[i,6]);
        if p==0 and q==0:
            out[i]=1.;continue
        ac=acf_lowarma(p,q,fitted[i,9],fitted[i,10],fitted[i,11],fitted[i,12],m)
        out[i]=qmc_entropy(ac,m,u,lookup)
    return out

def lookup_and_perms(m):
    perms=np.array(list(itertools.permutations(range(m))),dtype=np.int32)
    lookup=np.full(8**m,-1,dtype=np.int32)
    for idx,perm in enumerate(perms):
        # perm: sorted time indices; translate into ranks of each time position
        rank=np.empty(m,dtype=np.int32)
        for j,pos in enumerate(perm):rank[pos]=j
        code=sum(int(rank[i])<<(3*i) for i in range(m))
        lookup[code]=idx
    return lookup,perms

def run(bearings,dimensions,bench=False):
    set_num_threads(5)
    manifest=pd.read_csv(ROOT+'/ims_set2_b1_manifest.csv')
    datearr=pd.to_datetime(manifest.fecha_hora).to_numpy()
    raw1=np.load(ROOT+'/ims_set2_b1_samples.npy',mmap_mode='r')
    raw234=np.load(ROOT+'/IMS_Set2_channels234_samples.npy',mmap_mode='r')
    for m in dimensions:
        N=5*math.factorial(m);wn=20480//N
        lookup,perms=lookup_and_perms(m)
        if m>=5:
            K={5:4096,6:8192}[m]
            U=norm.ppf(qmc.Sobol(d=m,scramble=True,seed=20480+m).random_base2(int(np.log2(K))))
            U=np.asarray(U,dtype=np.float64)
        print(f'== m={m}, N={N}, n/windows/file={wn}, Hlin={"exact" if m==4 else "QMC"} ==',flush=True)
        for bearing in bearings:
            tic=time.monotonic()
            source=raw1 if bearing==1 else raw234[:,:,bearing-2]
            X=np.ascontiguousarray(source[:,:wn*N].reshape(-1,N))
            if bench:X=X[:min(1000,len(X))]
            vals=run_fit(X,m,lookup)
            print(f'BEARING {bearing} fit done {len(vals)} at {time.monotonic()-tic:.1f}s',flush=True)
            if m==4:H=all_hlin4(vals,perms)
            else:H=all_hlin_qmc(vals,m,lookup,U)
            print(f'BEARING {bearing} Hlin done {time.monotonic()-tic:.1f}s',flush=True)
            cols=['PE','ordinal_valid','ordinal_ties','patterns_distinct','RMS','p','q','mu','sigma2_innov','phi1','phi2','theta1','theta2','BIC_CSS','CSS_MSE','LB_Q5','LB_Q10','status']
            d=pd.DataFrame(vals,columns=cols)
            d['Hlin']=H
            d.insert(0,'record_idx',np.arange(len(d))//wn+1)
            d.insert(1,'window_idx',np.arange(len(d))%wn+1)
            d.insert(2,'sample_start',(np.arange(len(d))%wn)*N)
            for lag in [5,10]:
                dof=lag-d['p']-d['q']
                d[f'LB_pvalue_{lag}']=chi2.sf(d[f'LB_Q{lag}'].values,dof.values)
                d.loc[(dof<=0)|(d.status!=0),f'LB_pvalue_{lag}']=np.nan
            if not bench:
                d.insert(1,'fecha_hora',datearr[d.record_idx.to_numpy()-1])
                fn=f'{ROOT}/IMS_Set2_B{bearing}_m{m}_N{N}_ventanas.csv.gz'
                d.to_csv(fn,index=False,float_format='%.10g',compression={'method':'gzip','compresslevel':1})
                summary=d.groupby(['record_idx','fecha_hora']).agg(PE=('PE','mean'),Hlin=('Hlin','mean'),PE_valid=('PE','count'),Hlin_valid=('Hlin','count'),ties=('ordinal_ties','sum'),ord_patterns=('ordinal_valid','sum'),nAR2=('p',lambda x:int((x==2).sum())),LB5_pass=('LB_pvalue_5',lambda x:int((x>.05).sum()))).reset_index()
                summary['n_windows']=wn
                summary['fraction_ordinal_valid']=summary.ord_patterns/(wn*(N-m+1))
                summary['valid_record']=summary.fraction_ordinal_valid>=0.9
                summary.to_csv(f'{ROOT}/IMS_Set2_B{bearing}_m{m}_N{N}_resumen.csv',index=False)
                # simple quality-selected correlations excluding massively tied files
                good=summary[summary.valid_record].dropna(subset=['PE','Hlin'])
                rr=good.PE.corr(good.Hlin)
                dd=good.PE.diff().corr(good.Hlin.diff())
                print(f'  bearing={bearing}: models={int((d.status==0).sum())}; PE={int(d.PE.notna().sum())}; Hlin={int(d.Hlin.notna().sum())}; valid_records={len(good)}; r={rr:.4f}; first-difference-r={dd:.4f}',flush=True)
            else:
                print('head:\n',d[['PE','p','q','phi1','phi2','theta1','theta2','Hlin']].head().to_string(index=False),'corr',d.PE.corr(d.Hlin),flush=True)
            print('bearing',bearing,'elapsed',time.monotonic()-tic,flush=True)
            del X,vals,d

if __name__=='__main__':
    args=sys.argv[1:]
    bearings=[int(x) for x in next((s.split('=')[1] for s in args if s.startswith('b=')),'1,2,3,4').split(',')]
    dimensions=[int(x) for x in next((s.split('=')[1] for s in args if s.startswith('m=')),'4,5,6').split(',')]
    run(bearings,dimensions,bench=('bench' in args))
