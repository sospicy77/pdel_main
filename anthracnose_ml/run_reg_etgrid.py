import sys; sys.path.insert(0,'/home/user/pdel_main/anthracnose_ml')
import numpy as np, itertools
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import common as C
from run_reg_tune import data
dd,X,y=data(sys.argv[1],'A'); z=np.log10(y)
cv=list(RepeatedKFold(n_splits=5,n_repeats=10,random_state=3).split(X))
grid=list(itertools.product([7,10,12],[1,2,3,5],['sqrt',0.5,0.8]))
def one(i,tr,te):
    rk=C.rank_features(X.iloc[tr],z[tr],'reg',i).index; out=[]
    for k,leaf,mf in grid:
        cols=list(rk[:k]); m=make_pipeline(SimpleImputer(strategy='median'),ExtraTreesRegressor(500,min_samples_leaf=leaf,max_features=mf,random_state=i,n_jobs=1)).fit(X.iloc[tr][cols],z[tr])
        out.append(m.predict(X.iloc[te][cols]))
    return te,out
res=Parallel(n_jobs=4)(delayed(one)(i,tr,te) for i,(tr,te) in enumerate(cv))
for g,cfg in enumerate(grid):
    P=np.zeros((10,len(y)))
    for i,(te,out) in enumerate(res): P[i//5,te]=out[g]
    r=[r2_score(z,p) for p in P]; print(cfg,'%.3f ± %.3f'%(np.mean(r),np.std(r)))
