"""External validation on TCGA-BRCA (a completely different study from METABRIC).

Downloads the public TCGA breast cancer clinical file from cBioPortal's datahub (GitHub, git-lfs),
maps its columns onto the model's inputs and checks how well the model trained on METABRIC ranks
TCGA patients. Writes artifacts/external_validation.json.

Run:  python external_validation.py        (needs internet the first time)

Notes: TCGA has no tumor grade, mutation count, chemotherapy or hormone-therapy field, so those inputs are
missing; tumor size is approximated from the T stage (T1=15, T2=35, T3=65, T4=80 mm) and positive nodes
from the N stage / H&E node count. TCGA follow-up is short (median about 29 months).
"""
import sys, urllib.request
from pathlib import Path
HERE = Path(__file__).resolve().parent
D = str(HERE / "data" / "tcga") + "/"
(HERE / "data" / "tcga").mkdir(parents=True, exist_ok=True)
_URL = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_tcga/data_clinical_patient.txt"
if not (HERE / "data" / "tcga" / "old_data_clinical_patient.txt").exists():
    urllib.request.urlretrieve(_URL, D + "old_data_clinical_patient.txt")
import sys,json,warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,str(HERE))
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv
from app.features import *
import app.ml as ml
# ---------- TCGA -> METABRIC coding
p=pd.read_csv(D+'old_data_clinical_patient.txt',sep='\t',comment='#').replace({'[Not Available]':np.nan,'[Not Applicable]':np.nan,'[Not Evaluated]':np.nan,'[Discrepancy]':np.nan,'[Unknown]':np.nan})
p=p[p.SEX=='Female']
p['m']=pd.to_numeric(p.OS_MONTHS,errors='coerce'); p=p[p.m.notna()&(p.m>0)]
T=pd.DataFrame(index=p.index)
T['Age at Diagnosis']=pd.to_numeric(p.AGE,errors='coerce')
T['Inferred Menopausal State']=p.MENOPAUSE_STATUS.map(lambda v: 'Post' if isinstance(v,str) and v.startswith('Post') else 'Pre' if isinstance(v,str) and v.startswith('Pre') else np.nan)
st=p.AJCC_PATHOLOGIC_TUMOR_STAGE.astype(str)
T['Tumor Stage']=np.where(st.str.startswith('Stage IV'),4,np.where(st.str.startswith('Stage III'),3,np.where(st.str.startswith('Stage II'),2,np.where(st.str.startswith('Stage I'),1,np.nan))))
pt=p.AJCC_TUMOR_PATHOLOGIC_PT.astype(str)
T['Tumor Size']=np.where(pt.str.startswith('T1'),15.0,np.where(pt.str.startswith('T2'),35.0,np.where(pt.str.startswith('T3'),65.0,np.where(pt.str.startswith('T4'),80.0,np.nan))))
pn=p.AJCC_NODES_PATHOLOGIC_PN.astype(str)
he=pd.to_numeric(p.LYMPH_NODES_EXAMINED_HE_COUNT,errors='coerce')
pn_mid=np.where(pn.str.startswith('N0'),0.0,np.where(pn.str.startswith('N1'),2.0,np.where(pn.str.startswith('N2'),6.0,np.where(pn.str.startswith('N3'),12.0,np.nan))))
T['Lymph nodes examined positive']=np.where(np.isnan(pn_mid),np.nan,np.where(pn_mid==0,0.0,np.where(he.notna()&(he>0),he,pn_mid)))
T['Neoplasm Histologic Grade']=np.nan; T['Mutation Count']=np.nan
def pn_(v): return v if v in('Positive','Negative') else np.nan
T['ER Status']=p.ER_STATUS_BY_IHC.map(pn_); T['PR Status']=p.PR_STATUS_BY_IHC.map(pn_)
her=p.IHC_HER2.map(pn_); fish=p.HER2_FISH_STATUS.map(pn_); T['HER2 Status']=her.where(her.notna(),fish)
d=p.HISTOLOGICAL_DIAGNOSIS.astype(str)
T['Cancer Type Detailed']=np.where(d.str.contains('Ductal',case=False)&~d.str.contains('Lobular',case=False),'Invasive Ductal',np.where(d.str.contains('Lobular',case=False)&~d.str.contains('Ductal',case=False),'Invasive Lobular',np.where(d.str.contains('Mixed',case=False),'Mixed Ductal and Lobular','Other')))
T['Chemotherapy']=np.nan; T['Hormone Therapy']=np.nan
T['Radio Therapy']=p.RADIATION_TREATMENT_ADJUVANT.map({'YES':'Yes','NO':'No'})
sg=p.SURGICAL_PROCEDURE_FIRST.astype(str)
T['Type of Breast Surgery']=np.where(sg.str.contains('Lumpectomy|Quadrantectomy',case=False),'Breast Conserving',np.where(sg.str.contains('Mastectomy',case=False),'Mastectomy','')) ; T['Type of Breast Surgery']=pd.Series(T['Type of Breast Surgery'].to_numpy(),index=T.index).replace('',np.nan)
T=T[MODEL_INPUTS]
for c in CATEGORICAL_INPUTS: T[c]=T[c].astype(object).where(T[c].notna(),np.nan)
mon=np.ceil(p.m.to_numpy()); ev=(p.OS_STATUS=='1:DECEASED').to_numpy()
yT=Surv.from_arrays(ev,np.maximum(mon,1))
R={'tcga_patients':int(len(T)),'tcga_deaths':int(ev.sum()),'tcga_median_followup_months':float(np.median(mon)),'tcga_followup_ge_60m':int((mon>=60).sum()),
   'availability':{c:round(float(T[c].notna().mean()),2) for c in MODEL_INPUTS}}
print(json.dumps(R,indent=1))
# ---------- METABRIC
X,months,died,info=load_survival_frame(str(HERE/'data'/'Breast_Cancer_METABRIC_Epic_Hospital.csv'))
yM=Surv.from_arrays(died,months)
def cboot(y,risk,n=500,seed=0):
    rs=np.random.default_rng(seed); c0=concordance_index_censored(y['event'],y['time'],risk)[0]; cs=[]
    for _ in range(n):
        i=rs.integers(0,len(y),len(y))
        if y['event'][i].sum()<5: continue
        cs.append(concordance_index_censored(y['event'][i],y['time'][i],risk[i])[0])
    return [round(float(c0),3),round(float(np.percentile(cs,2.5)),3),round(float(np.percentile(cs,97.5)),3)]
def pipe(num,cat,est,scale=False):
    pre=ColumnTransformer([('n',Pipeline([('i',SimpleImputer(strategy='median'))]+([('s',StandardScaler())] if scale else [])),num),
                           ('c',Pipeline([('i',SimpleImputer(strategy='most_frequent')),('o',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
    return Pipeline([('p',pre),('m',est)])
res={}
# A. deployed model (needs all 15 inputs; missing ones get median/mode)
art=ml.SurvivalModel.load()
rowsA=art.pipeline.predict(T)  # risk scores via the same pipeline
res['A_deployed_v2_missing_inputs_imputed']=cboot(yT,rowsA)
# B. common-feature models
num=['Age at Diagnosis','Tumor Size','Tumor Stage','Lymph nodes examined positive']; cat=['Inferred Menopausal State','Cancer Type Detailed','ER Status','PR Status','HER2 Status','Type of Breast Surgery']
Xs,Ts=X[num+cat],T[num+cat]
def ET(): return ExtraSurvivalTrees(n_estimators=300,min_samples_leaf=15,min_samples_split=30,max_features='sqrt',n_jobs=2,random_state=0)
mET=pipe(num,cat,ET()).fit(Xs,yM); rET=mET.predict(Ts); res['B_extra_trees_common_features']=cboot(yT,rET)
mCox=pipe(num,cat,CoxPHSurvivalAnalysis(alpha=10),scale=True).fit(Xs,yM); rCox=mCox.predict(Ts); res['B_cox_common_features']=cboot(yT,rCox)
# same-model internal CV on METABRIC for reference
cs=[]
for tr,te in StratifiedKFold(5,shuffle=True,random_state=1).split(Xs,died):
    m=pipe(num,cat,ET()).fit(Xs.iloc[tr],yM[tr]); cs.append(concordance_index_censored(yM['event'][te],yM['time'][te],m.predict(Xs.iloc[te]))[0])
res['B_internal_metabric_cv_cindex']=[round(float(np.mean(cs)),3),round(float(np.std(cs)),3)]
# simple benchmarks on TCGA
res['benchmark_stage_only']=cboot(yT,np.nan_to_num(T['Tumor Stage'].to_numpy(),nan=2))
res['benchmark_nodes_only']=cboot(yT,np.nan_to_num(T['Lymph nodes examined positive'].to_numpy(),nan=0))
res['benchmark_age_only']=cboot(yT,np.nan_to_num(T['Age at Diagnosis'].to_numpy(),nan=58))
# truncated C-index at 5 years & time-dependent AUC
ok=~np.isnan(rET)
for nm,r in (('ET',rET),('Cox',rCox)):
    try: res[f'C_ipcw_tau60_{nm}']=round(float(concordance_index_ipcw(yM,yT,r,tau=60)[0]),3)
    except Exception as e: res[f'C_ipcw_tau60_{nm}']=str(e)[:80]
    for yr in (3,5):
        try:
            tt=np.array([yr*12.0]); ymask=yT['time']<yM['time'].max()
            res[f'AUC_{yr}y_{nm}']=round(float(cumulative_dynamic_auc(yM,yT[ymask],r[ymask],tt)[0][0]),3)
        except Exception as e: res[f'AUC_{yr}y_{nm}']=str(e)[:80]
# calibration of 5-year risk, common-feature ET (raw), by tertile
S5=mET[:-1].transform(Ts); sf=mET.named_steps['m'].predict_survival_function(S5,return_array=True); tm=mET.named_steps['m'].unique_times_
i5=np.searchsorted(tm,60,side='right')-1; pred5=1-sf[:,i5]
from sksurv.nonparametric import kaplan_meier_estimator
grp=np.digitize(pred5,np.quantile(pred5,[1/3,2/3])); cal=[]
for g in range(3):
    m_=grp==g; t,s=kaplan_meier_estimator(yT['event'][m_],yT['time'][m_]); k=np.searchsorted(t,60,side='right')-1
    cal.append({'n':int(m_.sum()),'mean_predicted_5y_risk':round(float(pred5[m_].mean()),3),'observed_5y_risk_KM':round(float(1-s[k]),3) if k>=0 else None,'deaths':int(yT['event'][m_].sum())})
res['D_calibration_5y_tertiles']=cal
R['results']=res; json.dump(R,open(HERE/'artifacts'/'external_validation.json','w'),indent=1)
print(json.dumps(res,indent=1))
