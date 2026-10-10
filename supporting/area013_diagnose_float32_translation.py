"""Registered post-run diagnosis; does not replace F13-10 outputs."""
import argparse,sys,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--source-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args();sys.path.insert(0,str(Path(a.source_dir).resolve()))
from utils.kernel_quadrature import kq_compress,indicator_kernel
torch.set_num_threads(1);rng=np.random.default_rng(20261019);s=.75;lm=np.exp(s*s/2);ls=np.sqrt((np.exp(s*s)-1)*np.exp(s*s));grid=[(0,1),(0,.9),(.15,1),(.15,.9)];rows=[];maximum=dict(W32=0.,identity32=0.,W64=0.,identity64=0.)
def score(w,z,y):
 q=w.double();v=z.double();n=q/q.sum();h=indicator_kernel(v,v)
 return np.array([float(q@(v-y).abs()+q@v-2*50/49*(q@h@q)),float(n@(v-y).abs()+n@v-2/(1-n@n)*(n@h@n))])
for r in range(50):
 for t in range(5):
  y=float((rng.lognormal(0,s,size=1)[0]-lm)/ls);yy=float(np.float32(y))
  for j,(mu,scale) in enumerate(grid):
   x=torch.tensor(mu+scale*(rng.lognormal(0,s,size=500)-lm)/ls,dtype=torch.float32);seed=20261018+100000+r*80+t*16+j*4+3;torch.manual_seed(seed);w,z=kq_compress(x,100,50);before=score(w,z,yy);q=w.double();pred=10*(float(q.sum())-50/49*(float(q.sum())**2-float(q@q)))
   f32=score(w,z+10,yy+10)-before;f64=score(w,z.double()+10,yy+10)-before;errs=dict(W32=abs(f32[1]),identity32=abs(f32[0]-pred),W64=abs(f64[1]),identity64=abs(f64[0]-pred))
   for k,v in errs.items():maximum[k]=max(maximum[k],v)
   merged=len(z)-len(torch.unique(z+10))
   if max(errs['W32'],errs['identity32'])>2e-4 or merged:
    rows.append(dict(r=r,t=t,j=j,seed=seed,node_count=len(z),merged_after_float32_shift=merged,min_spacing=float(torch.diff(z.sort().values).min()),errors=errs))
result=dict(disclosed_postrun_diagnosis=True,replayed_original_compressions=1000,affected_cases=rows,maximum_errors=maximum,hypothesis_supported=bool(rows and all(v['merged_after_float32_shift']>0 and max(v['errors']['W64'],v['errors']['identity64'])<=2e-4 for v in rows)),main_global_verdict_unchanged='FAIL',diagnosis_protocol_commit='9e96574ce4452542548686dd34d36c5044a8595a')
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
