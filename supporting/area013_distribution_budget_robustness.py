"""F13-10: unchanged source compression, prospectively registered robustness."""
import argparse,json,sys,subprocess,traceback,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import ndtr
from scipy.stats import t as student,lognorm,norm
from scipy.integrate import quad
p=argparse.ArgumentParser();p.add_argument('--source-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--protocol-commit',required=True);args=p.parse_args()
sys.path.insert(0,str(Path(args.source_dir).resolve()))
from utils.kernel_quadrature import kq_compress,indicator_kernel,error_terms_kq_comp
torch.set_num_threads(1)
configs=[(100,40,20),(200,40,20),(200,100,50),(500,100,50)];grid=[(0,1),(0,.9),(.15,1),(.15,.9)];names=['original','W','E','fair','empirical','sourcePWM'];families=['Normal','Lognormal','Student-t5'];R=50;T=5
s=.75;lm=np.exp(s*s/2);ls=np.sqrt((np.exp(s*s)-1)*np.exp(s*s));ts=np.sqrt(3/5);tg=2*quad(lambda x:student.cdf(x,5)*student.sf(x,5),0,np.inf,epsabs=1e-11,epsrel=1e-11)[0]
def sample(f,rng,n):
 if f==0:return rng.normal(size=n)
 if f==1:return (rng.lognormal(0,s,size=n)-lm)/ls
 return ts*rng.standard_t(5,size=n)
def reference(f,y,mu,a):
 v=(y-mu)/a
 if f==0:return a*(v*(2*ndtr(v)-1)+2*np.exp(-v*v/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))
 if f==1:
  q=lm+ls*v
  ae=lm-q if q<=0 else lm-q+2*(q*ndtr(np.log(q)/s)-lm*ndtr((np.log(q)-s*s)/s))
  return a/ls*(ae-lm*(2*ndtr(s/np.sqrt(2))-1))
 u=v/ts;ae=u*(2*student.cdf(u,5)-1)+2*(5+u*u)/4*student.pdf(u,5)
 return a*ts*(ae-tg)
def cdf(f,x,mu,a):
 v=(x-mu)/a
 if f==0:return norm.cdf(v)
 if f==1:return lognorm.cdf(lm+ls*v,s)
 return student.cdf(v/ts,5)
ref_checks=[]
for f in range(3):
 for y in [-2.,0.,2.]:
  for mu,a in grid:
   numeric=quad(lambda x:cdf(f,x,mu,a)**2,-np.inf,y,epsabs=1e-9,epsrel=1e-9)[0]+quad(lambda x:(1-cdf(f,x,mu,a))**2,y,np.inf,epsabs=1e-9,epsrel=1e-9)[0]
   ref_checks.append(dict(family=families[f],y=y,mu=mu,a=a,formula=float(reference(f,y,mu,a)),integral=numeric,error=abs(numeric-reference(f,y,mu,a))))
reference_pass=max(z['error'] for z in ref_checks)<=1e-6
print('Reference controls',reference_pass,flush=True)
def evaluate(w,z,y,m):
 q=w.double();v=z.double();n=q/q.sum();h=indicator_kernel(v,v);d=(v[:,None]-v[None,:]).abs()
 raw=q@(v-y).abs()+q@v-2*m/(m-1)*(q@h@q)
 W=n@(v-y).abs()+n@v-2/(1-n@n)*(n@h@n)
 E=n@(v-y).abs()-.5*(n@d@n)
 return np.array([float(raw),float(W),float(E)])
def full(x,y):
 v=x.double();v=v.sort().values;n=len(v);spread=((2*torch.arange(n,dtype=torch.float64)-n+1)*v).sum();a=(v-y).abs().mean();emp=float(a-spread/(n*n));fair=float(a-spread/(n*(n-1)))
 return np.array([fair,emp,emp+float(v.mean())/n])
cells=[];overall_start=time.perf_counter()
for f,family in enumerate(families):
 rng=np.random.default_rng(20261018+f);scores=np.full((4,R,T,4,2,6),np.nan);oracle=np.zeros((R,T,4));diags=[[] for _ in configs];failures=[[] for _ in configs];checks=[dict(public=0.,fair=0.,W=0.,E=0.,identity=0.,mass=0.,pair_fair=0.,negative=0.) for _ in configs];start=time.perf_counter()
 for r in range(R):
  for tt in range(T):
   y=float(sample(f,rng,1)[0]);y0=float(np.float32(y))
   for j,(mu,a) in enumerate(grid):
    oracle[r,tt,j]=reference(f,y,mu,a);allx=torch.tensor(mu+a*sample(f,rng,500),dtype=torch.float32)
    for ci,(M,rank,m) in enumerate(configs):
     seed=20261018+f*100000+r*80+tt*16+j*4+ci;x=allx[:M];parts=[];chk=checks[ci]
     try:
      for o,c in enumerate([0.,10.]):
       xx=x+c;yy=y0+c;torch.manual_seed(seed);w,z=kq_compress(xx,rank,m)
       if not torch.isfinite(w).all() or not torch.isfinite(z).all():raise ValueError('nonfinite nodes/weights')
       if len(torch.unique(z))!=len(z):raise ValueError('duplicate selected nodes')
       vals=evaluate(w,z,yy,m);tail=full(xx,yy);scores[ci,r,tt,j,o]=np.r_[vals,tail]
       if not np.isfinite(scores[ci,r,tt,j,o]).all():raise ValueError('nonfinite score')
       chk['mass']=max(chk['mass'],abs(float(w.sum())-1));chk['negative']=min(chk['negative'],float(w.min()))
       ids=(z[:,None]-xx[None,:]).abs().argmin(1).numpy();mapped=np.zeros(M);mapped[ids]=w.double().numpy()/float(w.double().sum());parts.append((w,z,vals,mapped,set(ids.tolist())))
       if r==0 and tt<2:
        torch.manual_seed(seed);pub=float(error_terms_kq_comp(xx,yy,n_low_rank=rank,n_compressed=m));chk['public']=max(chk['public'],abs(pub-vals[0]));v=xx.double();pair=float((v-yy).abs().mean()-(v[:,None]-v[None,:]).abs().sum()/(2*M*(M-1)));chk['pair_fair']=max(chk['pair_fair'],abs(pair-tail[0]))
      w,z,v,mp,ids=parts[0];fixed=evaluate(w,z+10,y0+10,m)-v;q=w.double();pred=10*(float(q.sum())-m/(m-1)*(float(q.sum())**2-float(q@q)))
      for key,value in [('identity',abs(fixed[0]-pred)),('W',abs(fixed[1])),('E',abs(fixed[2])),('fair',abs(scores[ci,r,tt,j,1,3]-scores[ci,r,tt,j,0,3]))]:chk[key]=max(chk[key],value)
      diags[ci].append(dict(r=r,t=tt,j=j,seed=seed,overlap=len(ids & parts[1][4]),support=[len(ids),len(parts[1][4])],weight_l1=float(abs(mp-parts[1][3]).sum()),shift=(scores[ci,r,tt,j,1]-scores[ci,r,tt,j,0]).tolist()))
     except Exception:failures[ci].append(dict(r=r,t=tt,j=j,seed=seed,error=traceback.format_exc()));scores[ci,r,tt,j]=np.nan
  if (r+1)%5==0:print(f'{family} dataset {r+1}/50 failures {[len(v) for v in failures]}',flush=True)
 for ci,config in enumerate(configs):
  sc=scores[ci];err=sc-oracle[:,:,:,None,None];block_mse=np.mean(err**2,axis=(1,2));summary={};diff={};boot=np.random.default_rng(20261019+f*4+ci).integers(0,R,size=(2000,R))
  for k,name in enumerate(names):
   summary[name]=[dict(bias=float(err[:,:,:,o,k].mean()),MAE=float(abs(err[:,:,:,o,k]).mean()),RMSE=float(np.sqrt((err[:,:,:,o,k]**2).mean()))) for o in range(2)]
   if k in [1,2,3]:
    bd=np.sqrt(block_mse[boot,:,k].mean(axis=1))-np.sqrt(block_mse[boot,:,0].mean(axis=1));diff[name]=[dict(estimate=summary[name][o]['RMSE']-summary['original'][o]['RMSE'],CI95=np.quantile(bd[:,o],[.025,.975]).tolist()) for o in range(2)]
  win=sc.mean(axis=1).argmin(axis=1);ow=oracle.mean(axis=1).argmin(axis=1);ranks={n:dict(oracle_disagreement=[float(np.mean(win[:,o,k]!=ow)) for o in range(2)],truth_selection=[float(np.mean(win[:,o,k]==0)) for o in range(2)],origin_winner_changes=int(np.sum(win[:,0,k]!=win[:,1,k]))) for k,n in enumerate(names)}
  chk=checks[ci];control=reference_pass and not failures[ci] and len(diags[ci])==1000 and chk['negative']>=0 and max(v for k,v in chk.items() if k!='pair_fair')<=2e-4 and chk['pair_fair']<=1e-9
  shifts=np.array([v['shift'] for v in diags[ci]]);median={n:float(np.median(abs(shifts[:,k]))) if len(shifts) else None for k,n in enumerate(names)};merit={n:bool(control and all(v['estimate']<0 and v['CI95'][1]<0 for v in diff[n])) for n in ['W','E','fair']}
  cells.append(dict(family=family,configuration=config,checks=chk,controls=bool(control),origin_signal=bool(control and median['original']>.01),correction_merit=merit,summary=summary,paired_RMSE_difference=diff,ranking=ranks,median_absolute_origin_shift=median,construction=dict(median_overlap=float(np.median([v['overlap'] for v in diags[ci]])),median_weight_l1=float(np.median([v['weight_l1'] for v in diags[ci]]))),failures=failures[ci],diagnostics=diags[ci],dataset_mse=block_mse.tolist(),dataset_mae=np.mean(abs(err),axis=(1,2)).tolist(),dataset_bias=err.mean(axis=(1,2)).tolist(),dataset_mean_scores=sc.mean(axis=1).tolist(),dataset_oracle_mean=oracle.mean(axis=1).tolist()))
 print(f'{family} completed in {time.perf_counter()-start:.1f}s',flush=True)
family_signal={f:sum(c['origin_signal'] for c in cells if c['family']==f) for f in families};clean=subprocess.check_output(['git','-C',args.source_dir,'diff','--stat'],text=True).strip();controls=not clean and all(c['controls'] for c in cells)
out=dict(protocol_commit=args.protocol_commit,source_commit=subprocess.check_output(['git','-C',args.source_dir,'rev-parse','HEAD'],text=True).strip(),source_diff=clean,reference_checks=ref_checks,controls=bool(controls),family_signal_cells=family_signal,robustness_retention=bool(controls and sum(family_signal.values())>=10 and min(family_signal.values())>=3),whole_grid_correction_merit={n:bool(controls and all(c['correction_merit'][n] for c in cells)) for n in ['W','E','fair']},runtime_seconds=time.perf_counter()-overall_start,cells=cells)
def sanitize(v):
 if isinstance(v,float) and not np.isfinite(v):return None
 if isinstance(v,list):return [sanitize(x) for x in v]
 if isinstance(v,dict):return {k:sanitize(x) for k,x in v.items()}
 return v
Path(args.output).write_text(json.dumps(sanitize(out),separators=(',',':')))
print(json.dumps({k:v for k,v in out.items() if k not in ['cells','reference_checks']},indent=2),flush=True)
