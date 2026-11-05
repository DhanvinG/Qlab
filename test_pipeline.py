import numpy as np
import pandas as pd
import cv2
import pytest
from pipeline import split_patients, regions, fit_labels, apply_labels, register, warp, eye_grades, safe_path


def test_split_never_separates_patient_eyes():
    m=pd.DataFrame({'patient_id':[str(i) for i in range(100)],'progression':[1]*80+[2]*20})
    s=split_patients(pd.concat([m,m]),42)
    assert len(s)==100 and pd.Series(s).value_counts().to_dict()=={'train':80,'val':10,'test':10}
    assert s==split_patients(m.sample(frac=1,random_state=3),42)


def test_followup_coverage_does_not_change_baseline_features():
    b=np.zeros((80,80),np.uint8);b[20:40,20:40]=4
    f=b.copy();v=b>0;mask=np.ones(b.shape,bool);partial=mask.copy();partial[:40]=False
    full=pd.DataFrame(regions(b,f,v,mask,mask,8))
    part=pd.DataFrame(regions(b,f,v,mask,partial,8))
    cols=[c for c in full if c.startswith('baseline_')]+['region','x','y']
    pd.testing.assert_frame_equal(full[cols],part[cols])
    assert not part.label_valid.all()


def test_label_fit_ignores_heldout_and_missing_targets():
    rng=np.random.default_rng(1);a=rng.normal(0,.02,(100,4))
    d=pd.DataFrame(a,columns=['change_'+l for l in ['CTW','EX','HE','MA']])
    d['key']=[str(i//5) for i in range(100)];d['split']='train';d['stable_eye']=True;d['label_valid']=True
    held=d.copy();held['split']='test';held.iloc[:,:4]=100
    p=fit_labels(d); assert p==fit_labels(pd.concat([d,held]))
    d.loc[0,'label_valid']=False
    scores,y=apply_labels(d,p);assert np.isnan(scores[0]) and np.isnan(y[0])
    same=d.copy();same.iloc[:,:4]=0
    assert np.nanmax(apply_labels(same,p)[1])==0
    with pytest.raises(ValueError):fit_labels(same)


def test_no_guessing_eye_grade():
    r=pd.Series({'key':'x','laterality':'','baseline_OS':1,'baseline_OD':3,'followup_OS':1,'followup_OD':4})
    with pytest.raises(ValueError):eye_grades(r)
    r['laterality']='OD';assert eye_grades(r)==(3,4)


def test_reject_path_escape(tmp_path):
    with pytest.raises(ValueError):safe_path(tmp_path,'../secret')


def test_known_affine_direction():
    rng=np.random.default_rng(12);b=np.zeros((512,512,3),np.uint8)
    cv2.circle(b,(256,256),230,(50,80,100),-1)
    for _ in range(700):
        x,y=rng.integers(75,437,2);v=int(rng.integers(90,220))
        cv2.circle(b,(int(x),int(y)),int(rng.integers(2,5)),(v,v,v),-1)
    forward=np.array([[1,0,8],[0,1,-5]],np.float32)
    f=warp(b,forward,b.shape[:2]).astype('uint8')
    M,_,_,qc=register(b,f)
    assert np.allclose(M,cv2.invertAffineTransform(forward),atol=.6)
    assert qc['overlap']>.9
