"""Fold-local survey preprocessing and grouped model comparison (never reads test.csv)."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
import importlib.metadata

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data_preprocessing'))
from encode_all import (encode_all, clean, raw_headers, EASE_COLUMN,
                        AGREEMENT_COLUMNS, AGREEMENT_MAP, SCHEMA)

CLASSES = ['Soccer', 'Baseball', 'Basketball']
OUT = ROOT / 'analysis/out/models'
GRIDS = {
    'softmax': {'C': [.001, .01, .1, 1., 10., 100.]},
    'forest': {'max_depth': [4, 8, 12, None], 'min_samples_leaf': [1, 3, 5],
               'max_features': ['sqrt', .5]},
    'mlp': {'hidden_layer_sizes': [(16,), (32,), (64,)], 'alpha': [.01, .1, 1., 10.]},
}


def sanitize(df):
    """Unknown structured answers become missing; preserve raw text/payment policy."""
    df = raw_headers(df).reset_index(drop=True)
    vals = pd.to_numeric(df[EASE_COLUMN].map(clean), errors='coerce')
    df.loc[~(vals.between(0, 10) & np.isfinite(vals)), EASE_COLUMN] = ''
    for col in AGREEMENT_COLUMNS:
        df.loc[~df[col].map(clean).isin(AGREEMENT_MAP), col] = ''
    for col, choices in SCHEMA.items():
        allowed = {clean(option) for option, _ in choices}
        def valid(value):
            selected = {p.strip() for p in clean(value).split(',') if p.strip()}
            return bool(selected) and not (selected - allowed)
        df.loc[~df[col].map(valid), col] = ''
    return df


class SurveyPreprocessor:
    """Train-only encoder, structured imputers/flags, log payment and optional scaling."""
    def _frame(self, raw, fitting):
        raw = sanitize(raw)
        encoded, artifact, _ = encode_all(raw, None if fitting else self.encoder_)
        if fitting:
            self.encoder_ = artifact
        features = encoded.drop(columns=['id', 'label'], errors='ignore').copy()
        for q, col in enumerate([EASE_COLUMN, *AGREEMENT_COLUMNS, *SCHEMA], 1):
            features[f'q{q}_missing'] = raw[col].map(clean).eq('').astype(float)
        if fitting:
            self.fill_ = {}
            for col in features:
                valid = features[col].dropna()
                if col in [EASE_COLUMN, *AGREEMENT_COLUMNS]:
                    value = valid.median() if len(valid) else 0.
                else:
                    modes = valid.mode()
                    value = modes.iloc[0] if len(modes) else 0.
                self.fill_[col] = float(value)
            self.columns_ = list(features)
        features = features.reindex(columns=self.columns_).fillna(self.fill_)
        features['payment_amount'] = np.log1p(features['payment_amount'])
        assert np.isfinite(features.to_numpy()).all()
        return features

    def fit(self, raw):
        features = self._frame(raw, True)
        self.scale_cols_ = [EASE_COLUMN, *AGREEMENT_COLUMNS, 'payment_amount']
        self.scaler_ = StandardScaler().fit(features[self.scale_cols_])
        return self

    def transform(self, raw, scaled=True):
        features = self._frame(raw, False)
        if scaled:
            features[self.scale_cols_] = self.scaler_.transform(features[self.scale_cols_])
        return features.to_numpy(dtype=float)


def text_sha256(path):
    """Hash tracked text consistently across LF and CRLF Git checkouts."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def provenance():
    files = ['data/splits/train.csv', 'data/splits/val.csv',
             'analysis/model_experiments.py', 'data_preprocessing/encode_all.py']
    return {'hash_format': 'sha256-lf-v1',
            'sha256': {p: text_sha256(ROOT/p) for p in files},
            'python': platform.python_version(),
            'packages': {p: importlib.metadata.version(p) for p in
                         ['numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib', 'threadpoolctl']},
            'seed': 311, 'folds': 5, 'test_accessed': False,
            'grids': json.loads(json.dumps(GRIDS)),
            'selection': 'CV mean accuracy, macro-F1, simplicity; validation accuracy, macro-F1, softmax then forest then mlp',
            'training_threads': 1}


def make_model(family, params, seed, expanded=False):
    if family == 'softmax':
        return LogisticRegression(solver='lbfgs', l1_ratio=0., tol=1e-4,
                                  max_iter=10000 if expanded else 2000, **params)
    if family == 'forest':
        return RandomForestClassifier(n_estimators=300, criterion='gini', bootstrap=True,
                                      class_weight=None, n_jobs=1, random_state=seed, **params)
    return MLPClassifier(activation='relu', solver='lbfgs', tol=1e-4,
                         max_iter=10000 if expanded else 2000,
                         max_fun=200000 if expanded else 50000, early_stopping=False,
                         random_state=seed, **params)


def fit_model(family, params, seed, X, y):
    messages = []
    start = time.perf_counter()
    for attempt in (0, 1):
        model = make_model(family, params, seed, expanded=bool(attempt))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with threadpool_limits(limits=1):
                model.fit(X, y)
        messages.extend(str(w.message) for w in caught)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        if converged:
            return model, messages, attempt, time.perf_counter()-start
    raise RuntimeError(f'{family} did not converge after retry: {messages}')


def metrics(y, pred):
    return {'accuracy': accuracy_score(y, pred),
            'macro_f1': f1_score(y, pred, labels=CLASSES, average='macro', zero_division=0)}


def simplicity(family, params):
    if family == 'softmax': return (params['C'],)
    if family == 'mlp': return (params['hidden_layer_sizes'][0], -params['alpha'])
    return (params['max_depth'] or 100000, -params['min_samples_leaf'],
            0 if params['max_features']=='sqrt' else params['max_features'])


def summarize(records):
    df = pd.DataFrame(records)
    return df.groupby(['family', 'config', 'params', 'stage'], sort=False).agg(
        cv_accuracy=('accuracy', 'mean'), cv_accuracy_sd=('accuracy', 'std'),
        cv_macro_f1=('macro_f1', 'mean'), cv_macro_f1_sd=('macro_f1', 'std'),
        train_accuracy=('train_accuracy', 'mean'), train_macro_f1=('train_macro_f1','mean'),
        mean_fit_seconds=('fit_seconds','mean'), retries=('retry','sum')).reset_index()


def select_best(records, configs, family):
    rows = summarize(records).query('family == @family').to_dict('records')
    return sorted(rows, key=lambda r: (-r['cv_accuracy'], -r['cv_macro_f1'],
                   simplicity(family, configs[r['config']])))[0]


def run(output=OUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prov = provenance()
    if (output/'run_manifest.json').exists():
        old = json.loads((output/'run_manifest.json').read_text())
        if old.get('provenance') != prov:
            raise RuntimeError('Existing results have different source/data/runtime hashes. Use a new --output directory.')
        if old.get('status') == 'complete':
            print('Verified existing completed experiment; no refitting or repeat validation.', flush=True)
            return output
        raise RuntimeError('Incomplete run exists. Choose a fresh --output directory; partial results are preserved.')
    manifest = {'status': 'running', 'provenance': prov, 'started_utc': pd.Timestamp.now(tz='UTC').isoformat()}
    (output/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    train = pd.read_csv(ROOT/'data/splits/train.csv', dtype=str, keep_default_na=False)
    y = train.label.to_numpy()
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=311).split(train, y, train.id))
    prepared, fold_rows, prep_audit = [], [], []
    for fold, (a, b) in enumerate(folds):
        assert set(train.iloc[a].id).isdisjoint(train.iloc[b].id)
        pre = SurveyPreprocessor().fit(train.iloc[a])
        prepared.append((pre.transform(train.iloc[a]), pre.transform(train.iloc[b]),
                         pre.transform(train.iloc[a], False), pre.transform(train.iloc[b], False), y[a], y[b]))
        for idx in b: fold_rows.append({'source_row': int(idx), 'id': train.iloc[idx].id, 'fold': fold})
        prep_audit.append({'fold': fold, 'train_rows': len(a), 'heldout_rows': len(b),
                          'train_groups': train.iloc[a].id.nunique(), 'heldout_groups': train.iloc[b].id.nunique(),
                          'features': len(pre.columns_),
                          'payment_mean': pre.encoder_['text_encoder']['payment_fill_value'],
                          'tfidf_terms': {k: len(v.vocabulary_) if v else 0 for k,v in pre.encoder_['text_encoder']['vectorizers'].items()}})
    pd.DataFrame(fold_rows).to_csv(output/'fold_assignments.csv', index=False)
    (output/'fold_preprocessing.json').write_text(json.dumps(prep_audit, indent=2))
    records, configs, boundary_notes = [], {}, []

    def evaluate(family, params, stage, seed=311, target=None, config=None):
        target = records if target is None else target
        if config is None:
            config = f'{family}_{len(configs):03d}'
            configs[config] = params
        for fold, data in enumerate(prepared):
            X, V = data[2:4] if family=='forest' else data[:2]
            ya, yb = data[4:]
            model, warns, retry, secs = fit_model(family, params, seed, X, ya)
            val_m = metrics(yb, model.predict(V)); tr_m = metrics(ya, model.predict(X))
            target.append({'family':family, 'config':config, 'params':json.dumps(params,sort_keys=True),
                           'stage':stage, 'seed':seed, 'fold':fold, **val_m,
                           'train_accuracy':tr_m['accuracy'], 'train_macro_f1':tr_m['macro_f1'],
                           'fit_seconds':secs, 'retry':retry, 'warnings':json.dumps(warns)})
        if target is records:
            pd.DataFrame(records).to_csv(output/'cv_folds.csv',index=False)
            summarize(records).to_csv(output/'cv_summary.csv',index=False)
        print(f'{family} {stage} {params}: CV accuracy {np.mean([r["accuracy"] for r in target[-5:]]):.4f}',flush=True)

    for family, grid in GRIDS.items():
        for params in ParameterGrid(grid): evaluate(family, params, 'initial')
    # One documented boundary expansion per applicable axis, using CV only.
    # A finite budget is explicit: this is not a guarantee of a global optimum.
    for family in GRIDS:
        best = select_best(records, configs, family)
        base = configs[best['config']]
        extensions = []
        if family=='softmax':
            if base['C'] in (.001,100.): extensions.append(('C',base['C']/10 if base['C']==.001 else base['C']*10))
        elif family=='mlp':
            if base['alpha'] in (.01,10.): extensions.append(('alpha',base['alpha']/10 if base['alpha']==.01 else base['alpha']*10))
            if base['hidden_layer_sizes'][0] in (16,64):
                extensions.append(('hidden_layer_sizes',(8,) if base['hidden_layer_sizes'][0]==16 else (128,)))
        else:
            if base['max_depth']==4: extensions.append(('max_depth',2))
            if base['min_samples_leaf']==5: extensions.append(('min_samples_leaf',10))
            if base['max_features']==.5: extensions.append(('max_features',1.))
        boundary_notes.append({'family':family,'initial_winner':base,'extensions':extensions,
            'rule':'One step beyond each selected expandable edge, varying one axis at a time; leaf size 1 and unlimited depth are natural boundaries.'})
        for key,value in extensions:
            evaluate(family, {**base,key:value}, 'boundary_extension')
    bests = {f:select_best(records, configs, f) for f in GRIDS}
    (output/'boundary_review.json').write_text(json.dumps(boundary_notes,indent=2))
    (output/'selected_configs.json').write_text(json.dumps({f:configs[r['config']] for f,r in bests.items()},indent=2))
    stability = []
    for family in ('forest','mlp'):
        best=bests[family]
        stability.extend([r for r in records if r['config']==best['config']])
        for seed in (312,313):
            evaluate(family,configs[best['config']],'stability',seed,stability,best['config'])
            pd.DataFrame(stability).to_csv(output/'seed_stability.csv',index=False)
    # External validation is first opened only after all tuning is complete.
    val=pd.read_csv(ROOT/'data/splits/val.csv',dtype=str,keep_default_na=False)
    assert set(train.id).isdisjoint(val.id)
    pre=SurveyPreprocessor().fit(train)
    comparison, predictions, reports = [], [], {}
    for family in GRIDS:
        params=configs[bests[family]['config']]
        X=pre.transform(train, family!='forest'); V=pre.transform(val, family!='forest')
        model,warns,retry,secs=fit_model(family,params,311,X,y)
        pred=model.predict(V)
        tr_m=metrics(y,model.predict(X)); va_m=metrics(val.label,pred)
        comparison.append({'family':family,'params':json.dumps(params,sort_keys=True),
                           **{k:bests[family][k] for k in ['cv_accuracy','cv_accuracy_sd','cv_macro_f1','cv_macro_f1_sd']},
                           'train_accuracy':tr_m['accuracy'],'train_macro_f1':tr_m['macro_f1'],
                           'validation_accuracy':va_m['accuracy'],'validation_macro_f1':va_m['macro_f1'],
                           'fit_seconds':secs,'retry':retry,'warnings':json.dumps(warns)})
        predictions.extend({'family':family,'source_row':i,'id':val.iloc[i].id,'actual':val.iloc[i].label,'predicted':p} for i,p in enumerate(pred))
        reports[family]=classification_report(val.label,pred,labels=CLASSES,output_dict=True,zero_division=0)
        pd.DataFrame(confusion_matrix(val.label,pred,labels=CLASSES),index=CLASSES,columns=CLASSES).to_csv(output/f'{family}_confusion.csv')
        joblib.dump({'preprocessor':pre,'model':model,'scaled':family!='forest','trained_on':'canonical train only'},output/f'{family}_development.joblib')
        print(f'{family}: validation accuracy {va_m["accuracy"]:.4f}, macro-F1 {va_m["macro_f1"]:.4f}',flush=True)
    cmp=pd.DataFrame(comparison)
    cmp.to_csv(output/'comparison.csv',index=False)
    pd.DataFrame(predictions).to_csv(output/'validation_predictions.csv',index=False)
    (output/'classification_reports.json').write_text(json.dumps(reports,indent=2))
    rank={'softmax':0,'forest':1,'mlp':2}
    winner=sorted(comparison,key=lambda r:(-r['validation_accuracy'],-r['validation_macro_f1'],rank[r['family']]))[0]['family']
    baseline=metrics(val.label,np.repeat('Soccer',len(val)))
    manifest.update(status='complete',completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                    selected_family=winner,baseline=baseline,cv_configurations=len(configs),
                    cv_fits=len(records),extra_seed_fits=20,validation_fits=3,
                    test_accessed=False,train_validation_refit=False)
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Completed; selected family:',winner,'; test set remains untouched.',flush=True)
    return output


if __name__=='__main__':
    # Avoid saving pickles referencing the ephemeral __main__ module.
    from model_experiments import run as execute
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    execute(parser.parse_args().output)
