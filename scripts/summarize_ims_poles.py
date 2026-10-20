#!/usr/bin/env python3
"""Generate record-level IMS Set 2 m=4 tables, including dominant reciprocal-root modulus.

Each output includes 982 valid records and their 24-record centered rolling means.
The average root modulus excludes windows fitted with p=0 (MA-only or white noise).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def one_file(path, out):
    d=pd.read_csv(path)
    needed={'record_idx','PE','Hlin','p','phi1','phi2','status'}
    if missing:=needed-set(d.columns):raise ValueError(f'{path}: missing {missing}')
    n=len(d)
    root=np.full(n,np.nan+0j,dtype=complex)
    p=d['p'].to_numpy();status=d['status'].to_numpy()
    i1=(p==1)&(status==0)
    root[i1]=d.loc[i1,'phi1'].to_numpy(float)
    i2=(p==2)&(status==0)
    if i2.any():
        a=d.loc[i2,'phi1'].to_numpy(float)
        b=d.loc[i2,'phi2'].to_numpy(float)
        disc=np.lib.scimath.sqrt(a*a+4*b)
        r1=(a+disc)/2;r2=(a-disc)/2
        take2=(abs(r2)>abs(r1)+1e-13)|((abs(abs(r2)-abs(r1))<=1e-13)&(r2.imag>r1.imag))
        r1[take2]=r2[take2]
        root[i2]=r1
    d['pole_mod']=abs(root)
    d.loc[~(i1|i2),'pole_mod']=np.nan
    # Data of user-approved figures: only first 982 valid acquisition records
    d=d[(d.record_idx>=1)&(d.record_idx<=982)]
    g=d.groupby('record_idx',sort=True).agg(
         PE=('PE','mean'),Hlin=('Hlin','mean'),pole_mod=('pole_mod','mean'),
         AR_windows=('pole_mod','count')).reset_index()
    g['symrelerr']=100*2*abs(g.PE-g.Hlin)/(abs(g.PE)+abs(g.Hlin))
    for c in ['PE','Hlin','pole_mod','symrelerr']:
        g[c+'_smooth']=g[c].rolling(24,center=True,min_periods=1).mean()
    if len(g)!=982:raise ValueError(f'{path}: expected 982 records, got {len(g)}')
    out=Path(out);out.parent.mkdir(exist_ok=True,parents=True)
    g.to_csv(out,index=False,float_format='%.12g')
    print(out, 'r_PE_Hlin=',round(g.PE.corr(g.Hlin),6))
    return g

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir',default='data/intermediate')
    ap.add_argument('--output-dir',default='data/regenerated_ims_record_tables')
    ap.add_argument('--bearings',default='1,2,3,4')
    a=ap.parse_args()
    for b in map(int,a.bearings.split(',')):
        one_file(Path(a.input_dir)/f'IMS_Set2_B{b}_m4_N120_ventanas.csv.gz',
                 Path(a.output_dir)/f'IMS_Set2_B{b}_m4_record_summaries_982.csv')
