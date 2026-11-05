"""Tianjin DR preprocessing. Run --help; see README.md for review protocol."""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LESIONS = ['CTW', 'EX', 'HE', 'MA']
CLINICAL = ['age', 'sex', 'hypertension', 'glucose', 'hba1c']
REPO = 'usama10/retinal-dr-longitudinal'
LESION_COMMIT = 'c1cc42416285ee486ad04c298d6d418c57e7e147'


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def pid(value):
    s = str(value).strip()
    if not re.fullmatch(r'\d+(\.0+)?', s): raise ValueError(f'Invalid patient identifier: {s}')
    return str(int(float(s))).zfill(5)


def safe_path(root, rel):
    p = (Path(root) / rel).resolve()
    if not p.is_relative_to(Path(root).resolve()): raise ValueError('Path outside dataset root')
    return p


def read_csv(path):
    return pd.read_csv(path, dtype={'patient_id': str, 'key': str})


def read_metadata(root):
    p = Path(root)/'Organized_Data of Patients.xlsx'
    b = pd.read_excel(p, sheet_name='Baseline')
    f = pd.read_excel(p, sheet_name='2 year follow-up')
    # Validate actual source headers before using their known positions.
    expected = {0:'ID',3:'OS Grade',4:'OD Grade',5:'At-risk Eye Grade',
                6:'sex',7:'age',8:'high-blood-pressure',23:'GLU',24:'HbA1c'}
    for i, prefix in expected.items():
        if i >= len(b.columns) or not str(b.columns[i]).startswith(prefix):
            raise ValueError(f'Workbook schema changed at baseline column {i}: expected {prefix}')
    for i, prefix in {0:'ID',1:'OS Grade',2:'OD Grade',3:'At-risk Eye Grade',4:'Progression'}.items():
        if not str(f.columns[i]).startswith(prefix): raise ValueError('Follow-up schema changed')
    b = b.iloc[:,[0,3,4,6,7,8,23,24]].copy()
    b.columns = ['patient_id','baseline_OS','baseline_OD','sex','age','hypertension','glucose','hba1c']
    f = f.iloc[:,[0,1,2,4]].copy()
    f.columns = ['patient_id','followup_OS','followup_OD','progression']
    for d in [b,f]:
        d['patient_id'] = d.patient_id.map(pid)
        if d.patient_id.duplicated().any(): raise ValueError('Duplicate metadata patient IDs')
        for col in d.columns[1:]: d[col] = pd.to_numeric(d[col], errors='coerce')
    return b.merge(f, on='patient_id', how='inner', validate='one_to_one')


def download(args):
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().dataset_info(REPO, revision=args.revision)
    snapshot_download(REPO, repo_type='dataset', revision=info.sha, local_dir=args.data)
    save_json(Path(args.data)/'download_provenance.json', {'repo':REPO,'revision':info.sha})
    print('Downloaded pinned dataset revision:', info.sha)


def split_patients(meta, seed):
    # Unknown outcomes stay their own stratum; never recode them to no progression.
    m = meta.sort_values('patient_id').drop_duplicates('patient_id')
    labels = m.progression.fillna(-1).astype(int)
    tr, tmp = train_test_split(m.patient_id, test_size=.2, random_state=seed, stratify=labels)
    lookup = dict(zip(m.patient_id, labels))
    va, te = train_test_split(tmp, test_size=.5, random_state=seed,
                             stratify=[lookup[x] for x in tmp])
    return {p:s for s, ids in [('train',tr),('val',va),('test',te)] for p in ids}


def audit(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if (out/'pairs.csv').exists(): raise ValueError('Use a new output folder; audit never overwrites an existing split')
    mf = Path(args.data)/'corrected_manifest.csv'
    d = read_csv(mf)
    required = {'key','patient_id','baseline_path','followup_path','quality'}
    if not required <= set(d.columns): raise ValueError(f'Missing manifest columns: {required-set(d.columns)}')
    d.patient_id = d.patient_id.map(pid)
    if d.key.duplicated().any(): raise ValueError('Duplicate pair keys')
    if not d.key.str.fullmatch(r'[A-Za-z0-9_-]+').all(): raise ValueError('Unsafe pair key')
    for col in ['baseline_path','followup_path']:
        if d[col].duplicated().any(): raise ValueError(f'Reused image path in {col}')
    records, seen, duplicate_records = [], {}, []
    for r in d.to_dict('records'):
        reasons=[]
        for col in ['baseline_path','followup_path']:
            p = safe_path(args.data, r[col])
            if not p.is_file(): reasons.append('missing_'+col); continue
            img=cv2.imread(str(p))
            if img is None: reasons.append('unreadable_'+col); continue
            h=sha(p); r[col+'_sha256']=h
            if h in seen:
                duplicate_records.append({'key':r['key'],'other_key':seen[h], 'sha256':h})
            seen[h]=r['key']
        r['audit_error']=';'.join(reasons); records.append(r)
    d=pd.DataFrame(records)
    pd.DataFrame(duplicate_records,columns=['key','other_key','sha256']).to_csv(out/'duplicates.csv',index=False)
    # Exclude all exact duplicate participants, not only the second occurrence.
    duplicate_keys={x[k] for x in duplicate_records for k in ['key','other_key']}
    d.loc[d.key.isin(duplicate_keys),'audit_error'] += ';duplicate_image'
    d.to_csv(out/'audit.csv',index=False)
    meta=read_metadata(args.data)
    d=d.merge(meta,on='patient_id',how='left',validate='many_to_one',indicator=True)
    d['metadata_present']=d['_merge'].eq('both');d=d.drop(columns='_merge')
    d.loc[~d.metadata_present,'audit_error'] += ';missing_metadata'
    d.to_csv(out/'audit.csv',index=False)
    good=d[d.audit_error.eq('')].copy()
    if good.empty: raise ValueError('No readable unique pairs with metadata; see audit.csv')
    try: splits=split_patients(good[['patient_id','progression']],args.seed)
    except ValueError as e: raise ValueError('Not enough patients per stratum for 80/10/10 split. Audit the full dataset, not a tiny pilot.') from e
    good['split']=good.patient_id.map(splits)
    good.to_csv(out/'pairs.csv',index=False)
    review=good[['key','patient_id','split']].copy()
    for c in ['laterality','same_eye','registration_ok','segmentation_ok','reviewer','notes']: review[c]=''
    review.to_csv(out/'review.csv',index=False)
    save_json(out/'audit_provenance.json',{'dataset':REPO,'manifest_sha256':sha(mf),
        'workbook_sha256':sha(Path(args.data)/'Organized_Data of Patients.xlsx'),
        'seed':args.seed,'source_pairs':len(d),'usable_pairs':len(good),
        'patients':good.patient_id.nunique(),'split_patients':pd.Series(splits).value_counts().to_dict()})
    print('Audit complete:',len(good),'pairs;',good.patient_id.nunique(),'patients')


def load_image(p, width):
    bgr=cv2.imread(str(p))
    if bgr is None: raise ValueError(f'Unreadable image: {p}')
    h,w=bgr.shape[:2]
    return cv2.resize(bgr,(width,round(h*width/w)),interpolation=cv2.INTER_AREA)


def fov(img):
    mask=(img.max(axis=2)>12).astype('uint8')
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours: raise ValueError('No retinal field of view')
    mask[:]=0;cv2.drawContours(mask,[max(contours,key=cv2.contourArea)],-1,1,-1)
    mask=cv2.erode(mask,np.ones((7,7),np.uint8))
    if mask.mean()<.15: raise ValueError('Very small field of view')
    return mask.astype(bool)


def warp(a, M, shape, nearest=False):
    return cv2.warpAffine(a.astype(np.float32),M,(shape[1],shape[0]),
                         flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,borderValue=0)


def register(base, follow):
    """Return FOLLOWUP -> BASELINE affine. Residuals are QC proxies, not true TRE."""
    bm,fm=fov(base),fov(follow)
    clahe=cv2.createCLAHE(2.,(8,8)); bg=clahe.apply(base[:,:,1]);fg=clahe.apply(follow[:,:,1])
    sift=cv2.SIFT_create(nfeatures=5000)
    kb,db=sift.detectAndCompute(bg,bm.astype('uint8')*255)
    kf,df=sift.detectAndCompute(fg,fm.astype('uint8')*255)
    if db is None or df is None: raise ValueError('No SIFT descriptors')
    matches=cv2.BFMatcher().knnMatch(df,db,k=2)
    good=[m[0] for m in matches if len(m)==2 and m[0].distance<.7*m[1].distance]
    if len(good)<16: raise ValueError('Too few distinctive correspondences')
    src=np.float32([kf[m.queryIdx].pt for m in good]);dst=np.float32([kb[m.trainIdx].pt for m in good])
    M,inside=cv2.estimateAffine2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=4.,maxIters=5000,confidence=.999)
    if M is None: raise ValueError('RANSAC failed')
    keep=inside.ravel().astype(bool)
    if keep.sum()<12 or keep.mean()<.25: raise ValueError('Weak RANSAC support')
    if cv2.contourArea(cv2.convexHull(dst[keep]))/bm.sum()<.08: raise ValueError('Matches concentrated in a small area')
    # ECC refines a prewarped image. ECC returns BASELINE -> PREWARPED;
    # invert that correction before composing with FOLLOWUP -> BASELINE.
    aligned=warp(fg,M,bg.shape)
    common=bm & (warp(fm,M,bg.shape,True)>.5)
    ecc_used=False;ecc_value=None
    try:
        score,C=cv2.findTransformECC(bg.astype('float32')/255,aligned/255,
            np.eye(2,3,dtype='float32'),cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,100,1e-5),
            inputMask=common.astype('uint8')*255,gaussFiltSize=5)
        candidate=(np.vstack([cv2.invertAffineTransform(C),[0,0,1]]) @ np.vstack([M,[0,0,1]]))[:2]
        old=np.linalg.norm(cv2.transform(src[None],M)[0]-dst,axis=1)[keep]
        new=np.linalg.norm(cv2.transform(src[None],candidate)[0]-dst,axis=1)[keep]
        if np.median(new)<=max(2.,np.median(old)*1.5): M=candidate;ecc_used=True;ecc_value=float(score)
    except cv2.error: pass
    singular=np.linalg.svd(M[:,:2],compute_uv=False)
    if np.linalg.det(M[:,:2])<=0 or min(singular)<.65 or max(singular)>1.5:
        raise ValueError('Implausible affine scale or reflection')
    common=bm & (warp(fm,M,bg.shape,True)>.5)
    overlap=common.sum()/bm.sum()
    residual=np.linalg.norm(cv2.transform(src[None],M)[0]-dst,axis=1)[keep]
    if overlap<.8 or np.percentile(residual,90)>20: raise ValueError('Registration QC failed')
    return M,bm,common,{'overlap':float(overlap),'inliers':int(keep.sum()),
        'median_residual_px':float(np.median(residual)),'p90_residual_px':float(np.percentile(residual,90)),
        'ecc_used':ecc_used,'ecc':ecc_value}


class Segmenter:
    def __init__(self,device, existing=None):
        import torch
        import segmentation_models_pytorch as smp
        from huggingface_hub import HfApi, hf_hub_download
        from fundus_lesions_toolkit.models import segment
        import fundus_lesions_toolkit.models.segmentation as lesion_module
        from fundus_lesions_toolkit.models.hf_hub import HuggingFaceModel
        from fundus_lesions_toolkit.constants import LESIONS as actual
        if list(actual)!=['BG']+LESIONS: raise ValueError('Lesion class order changed')
        self.torch,self.device,self.lesion_segment=torch,device,segment
        repo='gabriel-lepetitaimon/FundusVessel'
        existing=existing or {}
        revision=existing.get('vessel_revision') or HfApi().model_info(repo).sha
        checkpoint=hf_hub_download(repo,'Segmentation/resnet34.pt',revision=revision)
        self.vessel=smp.Unet('resnet34',encoder_weights=None,classes=2,activation='sigmoid')
        self.vessel.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
        self.vessel.to(device).eval()
        lesion_repo='ClementP/fundus-lesions-segmentation-unet_seresnext50_32x4d'
        lesion_revision=existing.get('lesion_revision') or HfApi().model_info(lesion_repo).sha
        self.lesion=HuggingFaceModel.from_pretrained(lesion_repo,revision=lesion_revision).model.to(device).eval()
        # Keep the author's preprocessing/inference, but freeze the model revision.
        lesion_module.get_model=lambda *a,**kw: self.lesion
        self.provenance={'vessel_repo':repo,'vessel_revision':revision,'vessel_sha256':sha(checkpoint),
                         'lesion_toolkit_commit':LESION_COMMIT,'lesion_model':'unet_seresnext50_32x4d',
                         'lesion_weights':'ALL','lesion_repo':lesion_repo,'lesion_revision':lesion_revision,'device':device}

    def __call__(self,bgr):
        t=self.torch
        # Lesion toolkit expects RGB uint8. Vessel author's load_img uses BGR [0,1].
        pred=self.lesion_segment(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),device=self.device,
                                 image_resolution=1024,reverse_autofit=True,compile=False)
        p=pred.detach().cpu().numpy()
        if p.shape != (5,*bgr.shape[:2]) or not np.isfinite(p).all():
            raise ValueError(f'Unexpected lesion probability shape/values: {p.shape}')
        if p.min() < -1e-5 or p.max()>1.00001: raise ValueError('Invalid lesion probabilities')
        labels=p.argmax(0).astype('uint8')
        h,w=bgr.shape[:2]; ph=(-h)%32;pw=(-w)%32
        x=t.from_numpy(bgr.copy()).permute(2,0,1)[None].float().to(self.device)/255
        x=t.nn.functional.pad(x,(pw//2,pw-pw//2,ph//2,ph-ph//2))
        with t.inference_mode(): v=self.vessel(x)[0,1,ph//2:ph//2+h,pw//2:pw//2+w].cpu().numpy()>.5
        mask=fov(bgr);labels[~mask]=0;v &= mask
        return labels,v


def regions(base_labels, follow_labels, vessels, bm, common, grid):
    """Baseline features use baseline FOV only. Outcomes use identical common pixels."""
    ys,xs=np.where(bm); yedges=np.linspace(ys.min(),ys.max()+1,grid+1,dtype=int)
    xedges=np.linspace(xs.min(),xs.max()+1,grid+1,dtype=int)
    records=[]
    for row in range(grid):
        for col in range(grid):
            s=np.s_[yedges[row]:yedges[row+1],xedges[col]:xedges[col+1]]
            b=bm[s];c=common[s]; n=int(b.sum()); nc=int(c.sum())
            if n==0 or n/b.size<.5: continue
            rec={'region':row*grid+col,'row':row,'col':col,'x':(col+.5)/grid,'y':(row+.5)/grid,
                 'baseline_pixels':n,'common_pixels':nc,'label_valid':bool(nc/n>=.8),
                 'baseline_vessel_density':float(vessels[s][b].mean())}
            for i,name in enumerate(LESIONS,1):
                rec['baseline_'+name]=float((base_labels[s][b]==i).mean())
                rec['change_'+name]=(float((follow_labels[s][c]==i).mean())-
                                     float((base_labels[s][c]==i).mean())) if nc else np.nan
            records.append(rec)
    return records


def panel(base,follow,M,bl=None,fl=None,grid=8):
    aligned=np.clip(warp(follow,M,base.shape[:2]),0,255).astype('uint8')
    mix=cv2.addWeighted(base,.5,aligned,.5,0)
    bm=fov(base); ys,xs=np.where(bm)
    for x in np.linspace(xs.min(),xs.max(),grid+1).astype(int): cv2.line(mix,(x,ys.min()),(x,ys.max()),(255,255,255),1)
    for y in np.linspace(ys.min(),ys.max(),grid+1).astype(int): cv2.line(mix,(xs.min(),y),(xs.max(),y),(255,255,255),1)
    tiles=[base,cv2.resize(follow,(base.shape[1],base.shape[0])),aligned,mix]
    if bl is not None:
        colors=np.array([[0,0,0],[0,190,255],[0,255,0],[255,120,0],[255,0,255]],dtype='uint8')
        tiles += [cv2.addWeighted(base,.6,colors[bl],.4,0),cv2.addWeighted(aligned,.6,colors[fl],.4,0)]
    tiles=[cv2.resize(x,(512,round(x.shape[0]*512/x.shape[1]))) for x in tiles]
    return np.concatenate(tiles,axis=1)


def process(args):
    out=Path(args.out); d=read_csv(out/'pairs.csv')
    if args.split!='all': d=d[d.split.eq(args.split)]
    d=d.sort_values('key')
    if args.limit: d=d.head(args.limit)
    folder=out/('registration' if args.registration_only else 'processed');folder.mkdir(exist_ok=True)
    config={'width':1024,'grid':args.grid,'device':args.device,'registration_only':args.registration_only,
            'pairs_sha256':sha(out/'pairs.csv'),'script_sha256':sha(__file__)}
    cp=folder/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config: raise ValueError('Configuration changed: use a new output directory')
    save_json(cp,config);segmenter=None
    for r in d.to_dict('records'):
        dest=folder/r['key'];dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists(): continue
        try:
            for col in ['baseline_path','followup_path']:
                if sha(safe_path(args.data,r[col])) != r[col+'_sha256']: raise ValueError('Source image changed since audit')
            b=load_image(safe_path(args.data,r['baseline_path']),1024)
            f=load_image(safe_path(args.data,r['followup_path']),1024)
            M,bm,common,qc=register(b,f)
            save_json(dest/'registration.json',dict(qc,affine_followup_to_baseline=M.tolist()))
            cv2.imwrite(str(dest/'review.jpg'),panel(b,f,M,grid=args.grid))
            if not args.registration_only:
                if segmenter is None:
                    mp=folder/'models.json'
                    segmenter=Segmenter(args.device,json.loads(mp.read_text()) if mp.exists() else None)
                    save_json(mp,segmenter.provenance)
                bl,v=segmenter(b);fl,_=segmenter(f)
                fl=warp(fl,M,b.shape[:2],True).astype('uint8')
                np.savez_compressed(dest/'masks.npz',baseline_lesions=bl,followup_lesions_aligned=fl,
                                    baseline_vessels=v,baseline_fov=bm,common_fov=common,affine=M)
                rows=regions(bl,fl,v,bm,common,args.grid)
                for x in rows: x.update(key=r['key'],patient_id=r['patient_id'],split=r['split'])
                pd.DataFrame(rows).to_csv(dest/'regions.csv',index=False)
                cv2.imwrite(str(dest/'review.jpg'),panel(b,f,M,bl,fl,args.grid))
                vessel_overlay=b.copy();vessel_overlay[v]=[255,255,255]
                cv2.imwrite(str(dest/'vessels.jpg'),vessel_overlay)
                cv2.imwrite(str(dest/'common_fov.png'),common.astype('uint8')*255)
            save_json(dest/'complete.json',{'key':r['key']})
            (dest/'error.txt').unlink(missing_ok=True)
            print(r['key'],'OK',flush=True)
        except Exception as e:
            (dest/'error.txt').write_text(f'{type(e).__name__}: {e}')
            print(r['key'],'FAILED:',e,flush=True)
    print('Review outputs:',folder)


def eye_grades(r):
    side=str(r.get('laterality','')).strip().upper()
    if side in ['OS','OD']: return r['baseline_'+side],r['followup_'+side]
    # Side not needed if BOTH eyes have identical grades at EACH visit.
    if r.baseline_OS==r.baseline_OD and r.followup_OS==r.followup_OD:
        return r.baseline_OS,r.followup_OS
    raise ValueError(f"{r['key']}: enter OS or OD in review.csv; asymmetric eye grades")


def fit_labels(reg, min_stable_eyes=10):
    cols=['change_'+l for l in LESIONS]
    stable=reg[(reg.split=='train') & reg.label_valid & reg.stable_eye]
    if stable.key.nunique()<min_stable_eyes: raise ValueError('Need at least 10 reviewed stable TRAINING eyes for threshold estimation')
    # SD of signed area-fraction changes; positive changes contribute to worsening.
    sd=stable[cols].std(ddof=1).to_numpy(float)
    if not np.isfinite(sd).all() or (sd<=1e-8).any():
        raise ValueError('A lesion has zero/undefined stable-training SD. Inspect segmentation; do not silently divide by epsilon.')
    s=(np.maximum(stable[cols].to_numpy(float),0)/sd).sum(axis=1)
    threshold=float(np.quantile(s,.95))
    return {'lesions':LESIONS,'sd':sd.tolist(),'threshold':threshold,
            'stable_training_eyes':int(stable.key.nunique()),'stable_training_regions':len(stable)}


def apply_labels(reg,params):
    a=reg[['change_'+l for l in LESIONS]].to_numpy(float)
    valid=reg.label_valid.to_numpy(bool)&np.isfinite(a).all(axis=1)
    scores=np.full(len(reg),np.nan); scores[valid]=(np.maximum(a[valid],0)/np.array(params['sd'])).sum(1)
    labels=np.full(len(reg),np.nan); labels[valid]=(scores[valid]>params['threshold']).astype(int)
    return scores,labels


def finalize(args):
    out=Path(args.out);d=read_csv(out/'pairs.csv');review=read_csv(out/'review.csv').fillna('')
    if review.key.duplicated().any(): raise ValueError('Duplicate review keys')
    d=d.merge(review.drop(columns=['patient_id','split'],errors='ignore'),on='key',how='left',validate='one_to_one')
    # Initial calibration requires a decided training cohort, not just a convenient pilot.
    if not (out/'final'/'label_parameters.json').exists():
        decided=d[['same_eye','registration_ok','segmentation_ok']].fillna('').apply(lambda c:c.astype(str).str.lower())
        rejected=decided.eq('no').any(axis=1)
        accepted=decided.eq('yes').all(axis=1) & d.reviewer.fillna('').astype(str).str.strip().ne('')
        pending=d.split.eq('train') & ~(rejected|accepted)
        if pending.any(): raise ValueError(f'{pending.sum()} training pairs still need review before threshold fitting')
        for k in d.loc[d.split.eq('train') & accepted,'key']:
            if not (out/'processed'/k/'complete.json').exists(): raise ValueError(f'Training pair {k} not processed')
    ok=np.ones(len(d),bool)
    for c in ['same_eye','registration_ok','segmentation_ok']: ok &= d[c].astype(str).str.lower().eq('yes')
    ok &= d.reviewer.fillna('').astype(str).str.strip().ne('')
    d=d[ok].copy();all_rows=[];excluded=[]
    for _,r in d.iterrows():
        p=out/'processed'/r['key']/'regions.csv'
        if not p.exists() or not (p.parent/'complete.json').exists(): continue
        bg,fg=eye_grades(r)
        if bg not in [1,2,3,4,5] or fg not in [1,2,3,4,5]:
            excluded.append({'key':r['key'],'reason':'invalid/treated/missing_eye_grade'});continue
        x=read_csv(p);x['stable_eye']=bool(bg==fg);x['baseline_grade']=bg
        for c in CLINICAL: x[c]=r[c]
        all_rows.append(x)
    if not all_rows: raise ValueError('No eligible reviewed pairs. Fill review.csv after inspecting images.')
    reg=pd.concat(all_rows,ignore_index=True)
    reg.label_valid=reg.label_valid.astype(str).str.lower().eq('true')
    folder=out/'final';folder.mkdir(exist_ok=True)
    # Freeze training-derived label parameters; never refit on val/test.
    pp=folder/'label_parameters.json'
    if pp.exists(): params=json.loads(pp.read_text())
    else:
        params=fit_labels(reg);save_json(pp,params)
    reg['change_score'],reg['worsening']=apply_labels(reg,params)
    reg.to_csv(folder/'regional_outcomes.csv',index=False)
    keys=['key','patient_id','split','region']
    features=keys+['row','col','x','y','baseline_pixels','baseline_vessel_density']+['baseline_'+l for l in LESIONS]+['baseline_grade']+CLINICAL
    X=reg[features].copy()
    # Impute clinical inputs on unique TRAINING patients, not region-weighted rows.
    ip=folder/'clinical_imputation.json'
    if ip.exists(): imp=json.loads(ip.read_text())
    else:
        train=X[X.split.eq('train')].drop_duplicates('patient_id')
        imp={}
        for c in CLINICAL:
            s=train[c].dropna()
            if s.empty: raise ValueError('No observed training values for '+c)
            imp[c]=float(s.mode().iloc[0] if c in ['sex','hypertension'] else s.median())
        save_json(ip,imp)
    for c in CLINICAL:
        X[c+'_missing']=X[c].isna().astype(int);X[c]=X[c].fillna(imp[c])
    X.to_csv(folder/'baseline_features.csv',index=False)
    reg[keys+['label_valid','change_score','worsening']].to_csv(folder/'targets.csv',index=False)
    pd.DataFrame(excluded,columns=['key','reason']).to_csv(folder/'excluded.csv',index=False)
    save_json(folder/'provenance.json',{'review_sha256':sha(out/'review.csv'),'pairs_sha256':sha(out/'pairs.csv'),
                                      'grid_config':json.loads((out/'processed'/'config.json').read_text()),
                                      'labels':'automatically derived; not clinical ground truth'})
    print('Final features and targets:',folder,'; rows:',len(reg))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['download','audit','process','finalize'])
    p.add_argument('--data',default='data');p.add_argument('--out',default='outputs')
    p.add_argument('--revision',default='main');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--grid',type=int,choices=[8,12,16],default=8)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--split',choices=['train','val','test','all'],default='train')
    p.add_argument('--limit',type=int,default=0);p.add_argument('--registration-only',action='store_true')
    a=p.parse_args();cv2.setRNGSeed(a.seed)
    globals()[a.command](a)

if __name__=='__main__': main()
