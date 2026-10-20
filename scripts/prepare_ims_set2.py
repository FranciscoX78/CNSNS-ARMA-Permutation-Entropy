#!/usr/bin/env python3
"""Convert 984 original NASA IMS Set 2 text snapshots to the exact array layout used by fit_ims_windows.py."""
import argparse
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

def run(raw, out):
    raw=Path(raw);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    files=[]
    for p in raw.rglob('*'):
        if not p.is_file():continue
        try: stamp=dt.datetime.strptime(p.name, '%Y.%m.%d.%H.%M.%S')
        except ValueError:continue
        files.append((stamp,p))
    files.sort()
    if len(files)!=984:raise ValueError(f'Expected 984 timestamped Set 2 snapshots, found {len(files)} under {raw}')
    x1=np.lib.format.open_memmap(out/'ims_set2_b1_samples.npy',mode='w+',dtype='float64',shape=(984,20480))
    x234=np.lib.format.open_memmap(out/'IMS_Set2_channels234_samples.npy',mode='w+',dtype='float64',shape=(984,20480,3))
    for i,(stamp,path) in enumerate(files):
        v=np.loadtxt(path,dtype=np.float64)
        if v.shape!=(20480,4):raise ValueError(f'{path}: expected shape (20480,4), got {v.shape}')
        x1[i]=v[:,0]
        x234[i]=v[:,1:4]
        if (i+1)%100==0:print('Converted',i+1,'/984',flush=True)
    x1.flush();x234.flush()
    pd.DataFrame({'record_idx':range(1,985),'fecha_hora':[s.isoformat(sep=' ') for s,_ in files],
                  'filename':[p.name for _,p in files]}).to_csv(out/'ims_set2_b1_manifest.csv',index=False)
    print('Created .npy matrices and manifest in',out)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--raw-dir',required=True,help='directory holding 984 Set 2 timestamped files')
    p.add_argument('--out-dir',default='data/intermediate')
    a=p.parse_args();run(a.raw_dir,a.out_dir)
