#!/usr/bin/env python3
"""Recalculate core manuscript statistics from committed derived data. No NASA raw download needed."""
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
A=pd.read_csv(ROOT/'data/experiment_A/CNSNS_synthetic_A_all_windows.csv')
B=pd.read_csv(ROOT/'data/experiment_B/CNSNS_synthetic_B_all_windows_AR1.csv')
assert len(A)==270 and len(B)==270
assert set(A.regime)==set(range(1,10))
assert set(B.regime)==set(range(1,10))
cor=A.PE_emp.corr(A.Hlin_fit_gaussian)
sre=100*np.mean(2*np.abs(A.PE_emp-A.Hlin_fit_matched_N)/(np.abs(A.PE_emp)+np.abs(A.Hlin_fit_matched_N)))
print('A Pearson corr PE/Hlin: ',format(cor,'.6f'))
print('A symmetric rel. error matched-N: ',format(sre,'.4f'),'%')
assert abs(cor-0.9989557)<1e-5
assert abs(sre-0.3835)<0.002
B['delta']=B.PE_emp-B.AR1_ref_mean
for a in [0,0.8,1.5,2]:
    print('B alpha',a,'mean gap AR1 matched-N',round(B[B.alpha.eq(a)].delta.mean(),6))
assert abs(B[B.alpha.eq(1.5)].delta.mean()-0.04549)<1e-4
for b in range(1,5):
    data=pd.read_csv(ROOT/f'data/ims/record_level_m4/IMS_Set2_B{b}_m4_record_summaries_982.csv')
    assert len(data)==982 and int(data.record_idx.min())==1 and int(data.record_idx.max())==982
    r=data.PE.corr(data.Hlin)
    print(f'IMS bearing {b}: records={len(data)}, r={r:.4f}')
    assert abs(r-[.935,.975,.721,.951][b-1])<.003
print('VERIFIED: key archived, derived numerical results.')
