"""Recompute saved sleep-model results without retraining or test-set tuning.

Run from the repository root using the command in "실행환경(Runtime Environment)/실행환경 안내.md".
Dependencies: numpy, pandas, matplotlib, joblib, scikit-learn==1.7.2.
Only deserialize model files supplied in the trusted team package.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'sleep_model_matplotlib'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import joblib
import sklearn
import scipy
from paths import (REPO, DATA_ROOT, DATA_DIR, RESULTS_DIR, CV_DIR, TEST_DIR, TREE_DIR,
                   ENV_DIR, FIGURE_DIR, MODEL_DIR, model_file)

FEATURES = ['past3_latency_median_min', 'past3_latency_n',
            'prev_sleep_interval_min', 'past3_sleep_interval_median_min',
            'past3_sleep_interval_n']
TARGETS = {'A': 'bed_to_sleep_min', 'B': 'sleep_quality'}


def metrics(df, task):
    error = df.predicted.to_numpy() - df.actual.to_numpy()
    absolute = pd.Series(np.abs(error), index=df.index)
    return {'n': len(df), 'people': df.participant_id.nunique(),
            'macro_mae': float(absolute.groupby(df.participant_id).mean().mean()),
            'mae': float(absolute.mean()), 'rmse': float(np.sqrt(np.mean(error**2))),
            'bias_pred_minus_actual': float(error.mean()),
            'within_tolerance': float((absolute <= (10 if task == 'A' else 1)).mean())}


def grouped_scores(df, grouping):
    rows = []
    for keys, g in df.groupby(grouping, observed=True, dropna=False):
        if not isinstance(keys, tuple): keys = (keys,)
        task = g.task.iloc[0]
        rows.append({**dict(zip(grouping, keys)), **metrics(g, task),
                     'actual_mean': float(g.actual.mean()),
                     'predicted_mean': float(g.predicted.mean())})
    return pd.DataFrame(rows)


def numpy_forward(saved, frame):
    x = saved['scaler'].transform(frame[saved['features']]).astype('float32')
    outputs = []
    for weights in saved['weights']:
        h = x
        for layer in ('0', '3'):
            h = np.maximum(h @ weights[layer + '.weight'].T + weights[layer + '.bias'], 0)
        h = (h @ weights['5.weight'].T + weights['5.bias']).ravel()
        if saved['task'] == 'A':
            outputs.append(np.logaddexp(np.float32(0), h))
        else:
            outputs.append(1 / (1 + np.exp(-h)))
    average = np.mean(outputs, axis=0)
    return average * 30 if saved['task'] == 'A' else 1 + average * 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO)
    args = parser.parse_args()
    output = args.output.resolve()
    out = output / RESULTS_DIR.name / '상세분석'
    figs = output / FIGURE_DIR.name
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                               'font.size': 11, 'axes.spines.top': False,
                               'axes.spines.right': False, 'savefig.facecolor': 'white',
                               'figure.facecolor': 'white', 'axes.titleweight': 'bold'})
    frames = {name: pd.read_csv(DATA_DIR / (name + '.csv')) for name in
              ('train_melidos_all', 'test_melidos', 'test_korea')}
    checks = []
    def check(name, detail):
        checks.append({'check': name, 'status': 'PASS', 'detail': detail})

    # Reconstruct prepared data from original, selected source records.
    code_dir = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('sleep_prepare', code_dir / 'prepare.py')
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    for name, original in frames.items():
        recreated = prep.extract(DATA_ROOT, 'train_melidos' if name == 'train_melidos_all' else name)
        pd.testing.assert_frame_equal(original, recreated, check_dtype=False, check_exact=False,
                                      rtol=1e-9, atol=1e-8)
        check('raw_to_prepared_' + name, f'{len(original)} records match stored prepared table')
    # Full original CSVs are published directly; reconstruction above reads them.
    full_team_zip = None
    verification_warnings = []
    ids = [set(df.participant_id) for df in frames.values()]
    assert len(set.union(*ids)) == sum(map(len, ids))
    check('split_participant_overlap', '0 participants shared across all three splits')
    for name, df in frames.items():
        dated = df.dropna(subset=['night_date'])
        assert not dated.duplicated(['participant_id', 'night_date']).any()
        for _, g in dated.groupby('participant_id'):
            g = g.sort_values('night_date')
            for row in g.itertuples():
                past = g[g.night_date < row.night_date]
                latency, duration = past.bed_to_sleep_min.dropna().tail(3), past.sleep_interval_min.dropna().tail(3)
                observed = [row.past3_latency_n, row.past3_latency_median_min,
                            row.past3_sleep_interval_n, row.past3_sleep_interval_median_min,
                            row.prev_sleep_interval_min]
                expected = [len(latency), latency.median(), len(duration), duration.median(),
                            past.sleep_interval_min.iloc[-1] if len(past) else np.nan]
                np.testing.assert_allclose(observed, expected, equal_nan=True, atol=1e-8)
        check('past_only_history_' + name, f'All {len(dated)} dated rows use strictly earlier dates only')

    scored = pd.read_csv(TEST_DIR / 'holdout_predictions.csv')
    tree_scored = pd.read_csv(TREE_DIR / 'holdout_predictions.csv').rename(columns={'prediction':'predicted'})
    all_metric_rows = []
    for source, scored_df in [('primary', scored), ('posthoc_tree', tree_scored)]:
        stored = pd.read_csv((TEST_DIR if source == 'primary' else TREE_DIR) / 'holdout_metrics.csv')
        grouping = ['task', 'dataset', 'role', 'model'] if source == 'primary' else ['task', 'dataset', 'model']
        for keys, g in scored_df.groupby(grouping):
            summary = {**dict(zip(grouping, keys)), **metrics(g, g.task.iloc[0]), 'source': source}
            ref = stored
            for col, value in zip(grouping, keys): ref = ref[ref[col] == value]
            assert len(ref) == 1
            for col in ('n', 'people', 'macro_mae', 'mae', 'rmse', 'bias_pred_minus_actual', 'within_tolerance'):
                np.testing.assert_allclose(summary[col], ref[col].iloc[0], atol=1e-9, rtol=1e-9)
            all_metric_rows.append(summary)
        check('recomputed_metrics_' + source, f'All {len(stored)} saved metric rows match recomputation')
    all_metrics = pd.DataFrame(all_metric_rows)
    all_metrics.to_csv(out / 'verified_metrics.csv', index=False, encoding='utf-8-sig')

    # Reconstruct PyTorch MLP inference mathematically with NumPy. No torch needed.
    datasets = {'melidos_holdout': frames['test_melidos'],
                'korea_last7': frames['test_korea'].query('is_korea_final7 == 1'),
                'korea_all': frames['test_korea']}
    inference, saved_models = [], {}
    for task in ('A', 'B'):
        for role in ('selected', 'mlp'):
            saved = joblib.load(model_file(task, role))
            saved_models[(task, role)] = saved
            training=frames['train_melidos_all'].dropna(subset=[TARGETS[task]] + (['sleep_interval_min'] if task=='B' else []))
            raw_x=training[saved['features']].to_numpy()
            medians=np.nanmedian(raw_x,axis=0)
            np.testing.assert_allclose(saved['scaler'].steps[0][1].statistics_,medians,atol=1e-8)
            filled=np.where(np.isnan(raw_x),medians,raw_x)
            scale_step=saved['scaler'].steps[1][1]
            np.testing.assert_allclose(scale_step.mean_,np.mean(filled,axis=0),atol=1e-8)
            np.testing.assert_allclose(scale_step.var_,np.var(filled,axis=0),atol=1e-8)
            for dataset, expected in scored[(scored.task == task) & (scored.role == role)].groupby('dataset'):
                keys = ['participant_id', 'site_code', 'night_date']
                inputs = expected[keys].merge(datasets[dataset], on=keys, validate='one_to_one', how='left')
                pred = numpy_forward(saved, inputs)
                gap = float(np.max(np.abs(pred - expected.predicted.to_numpy())))
                np.testing.assert_allclose(pred, expected.predicted, atol=1e-4, rtol=1e-6)
                inference.append({'task': task, 'role': role, 'dataset': dataset,
                                  'n': len(inputs), 'max_absolute_prediction_difference': gap})
    pd.DataFrame(inference).to_csv(out / 'numpy_model_verification.csv', index=False, encoding='utf-8-sig')
    check('numpy_saved_model_inference', f'{len(inference)} comparisons pass atol=0.0001, rtol=0.000001')
    check('saved_preprocessing_train_only', 'Saved imputer medians and scaler mean/variance match eligible TRAIN records for A/B')

    rng = np.random.default_rng(42)
    intervals = []
    for task, dataset, ref in [('A','melidos_holdout','baseline'), ('A','korea_last7','baseline'),
                               ('A','korea_all','baseline'), ('B','melidos_holdout','baseline'),
                               ('B','melidos_holdout','without_duration')]:
        d = scored[(scored.task == task) & (scored.dataset == dataset)].copy()
        d['ae'] = (d.predicted-d.actual).abs()
        p = d.groupby(['participant_id','role']).ae.mean().unstack()
        gap = (p.selected-p[ref]).to_numpy()
        boot = rng.choice(gap,(10000,len(gap)),replace=True).mean(axis=1)
        lo,hi = np.quantile(boot,[0.025,0.975])
        intervals.append({'task':task,'dataset':dataset,'comparison':ref,'delta_macro_mae':float(gap.mean()),
                          'lower_95':float(lo),'upper_95':float(hi),'people':len(gap)})
    ci = pd.DataFrame(intervals)
    original_ci = pd.read_csv(TEST_DIR / 'paired_intervals.csv')
    pd.testing.assert_frame_equal(ci, original_ci, check_exact=False, atol=1e-9, rtol=1e-9)
    ci.to_csv(out / 'verified_paired_intervals.csv', index=False, encoding='utf-8-sig')
    check('participant_bootstrap_10000', 'All stored primary 95% paired participant bootstrap intervals match')

    joined = []
    for dataset, g in scored[scored.role.isin(['selected','baseline'])].groupby('dataset'):
        joined.append(g.merge(datasets[dataset], on=['participant_id','site_code','night_date'],
                              validate='many_to_one', how='left'))
    detail = pd.concat(joined, ignore_index=True)
    detail['absolute_error'] = (detail.predicted-detail.actual).abs()
    detail[detail.role=='selected'].sort_values('absolute_error',ascending=False).groupby(['task','dataset'],group_keys=False).head(10)[
        ['task','dataset','participant_id','night_date','site_code','source_file','source_record_number','actual','predicted','error','absolute_error']
    ].to_csv(out/'largest_error_examples.csv',index=False,encoding='utf-8-sig')
    detail['latency_band'] = pd.cut(detail.actual, bins=[-np.inf,10,30,60,np.inf],
                                    labels=['0-10','10-30','30-60','60+'], right=True)
    detail['history_band'] = pd.cut(detail.past3_latency_n, bins=[-1,0,2,3],
                                    labels=['0 records','1-2 records','3 records'])
    latency = grouped_scores(detail[detail.task == 'A'], ['task','dataset','role','latency_band'])
    quality = grouped_scores(detail[detail.task == 'B'], ['task','dataset','role','actual'])
    history = grouped_scores(detail, ['task','dataset','role','history_band'])
    sites = grouped_scores(detail, ['task','dataset','role','site_code'])
    person = grouped_scores(detail, ['task','dataset','role','participant_id'])
    for name, df in [('latency_bands',latency),('quality_bands',quality),('history_bands',history),
                     ('site_errors',sites),('participant_errors',person)]:
        df.to_csv(out / (name+'.csv'), index=False, encoding='utf-8-sig')

    # Distribution and observed follow-up: span is last-date minus first-date plus one.
    distributions, followup, split_summary = [], [], []
    for split, frame in frames.items():
        for col in FEATURES + ['bed_to_sleep_min','sleep_interval_min','sleep_quality']:
            v = frame[col].dropna()
            distributions.append({'split':split,'variable':col,'n_valid':len(v),'n_missing':int(frame[col].isna().sum()),
                                  'mean':v.mean(),'std':v.std(),'min':v.min(),'q25':v.quantile(.25),
                                  'median':v.median(),'q75':v.quantile(.75),'max':v.max()})
        for pid, group in frame.groupby('participant_id'):
            dates = pd.to_datetime(group.night_date.dropna())
            followup.append({'split':split,'participant_id':pid,'records':len(group),'dated_records':len(dates),
                             'first_date':str(dates.min().date()),'last_date':str(dates.max().date()),
                             'span_days':int((dates.max()-dates.min()).days+1)})
        split_summary.append({'split':split,'records':len(frame),'people':frame.participant_id.nunique(),
                              'A_eligible':int(frame.bed_to_sleep_min.notna().sum()),
                              'B_eligible':int(frame[['sleep_quality','sleep_interval_min']].notna().all(axis=1).sum())})
    pd.DataFrame(distributions).to_csv(out / 'input_distributions.csv',index=False,encoding='utf-8-sig')
    followup = pd.DataFrame(followup)
    followup.to_csv(out / 'followup_by_participant.csv',index=False,encoding='utf-8-sig')
    followup_summary=followup.groupby('split').agg(
        records_min=('records','min'),records_median=('records','median'),records_max=('records','max'),
        span_days_min=('span_days','min'),span_days_median=('span_days','median'),span_days_max=('span_days','max'))
    followup_summary.to_csv(out/'followup_summary.csv',encoding='utf-8-sig')
    pd.DataFrame(split_summary).to_csv(out/'verified_split_summary.csv',index=False,encoding='utf-8-sig')
    cv = pd.read_csv(CV_DIR/'cv_summary.csv')
    best = cv.sort_values('macro_mae').groupby(['task','variant','model'],as_index=False).first()
    best.to_csv(out/'cv_best_by_algorithm.csv',index=False,encoding='utf-8-sig')
    ablation = best[(best.task=='B') & (best.model=='mlp')]
    ablation.to_csv(out/'b_same_mlp_ablation_cv.csv',index=False,encoding='utf-8-sig')
    byfold=pd.read_csv(CV_DIR/'cv_by_fold.csv')
    chosen_folds=[]
    for task in ('A','B'):
        g=byfold[(byfold.task==task)&(byfold.variant=='full')&(byfold.model=='mlp') &
                 byfold.params.str.contains('"width": 16') & byfold.params.str.contains('"epochs": 60')]
        chosen_folds.append({'task':task,'fold_macro_mae_unweighted_mean':float(g.macro_mae.mean()),
                             'fold_macro_mae_sample_sd':float(g.macro_mae.std(ddof=1)),
                             'pooled_oof_participant_macro_mae':float(best[(best.task==task)&(best.variant=='full')&(best.model=='mlp')].macro_mae.iloc[0]),
                             'folds':5})
    pd.DataFrame(chosen_folds).to_csv(out/'selected_mlp_cv_fold_summary.csv',index=False,encoding='utf-8-sig')
    curves=pd.read_csv(CV_DIR/'learning_curves.csv')
    curves.groupby(['task','seed']).agg(first_epoch=('epoch','min'),last_epoch=('epoch','max'),
        first_train_loss=('train_loss_scaled','first'),last_train_loss=('train_loss_scaled','last'),
        first_validation_macro_mae=('validation_macro_mae','first'),last_validation_macro_mae=('validation_macro_mae','last')).to_csv(out/'learning_curve_summary.csv',encoding='utf-8-sig')

    # Figures use complete saved evaluation rows, no trimming or outlier deletion.
    colors = {'selected':'#138C8F','baseline':'#D88432'}
    def save(fig,name):
        fig.savefig(figs/name,dpi=180,bbox_inches='tight')
        plt.close(fig)
    selected = detail[detail.role=='selected']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,dataset,title in zip(axes,['melidos_holdout','korea_last7'],['MeLiDos 별도 평가','국내 마지막 7일']):
        g=selected[(selected.task=='A')&(selected.dataset==dataset)]
        top=max(g.actual.max(),g.predicted.max())*1.03
        ax.scatter(g.actual,g.predicted,s=16,alpha=.4,c=colors['selected'],edgecolors='none')
        ax.plot([0,top],[0,top],color='#777777',ls='--',lw=1)
        ax.set(xlabel='실제 입면 지연 (분)',ylabel='MLP 예측 (분)',title=f'{title}  n={len(g)}',xlim=(0,top),ylim=(0,top))
    save(fig,'01_A_actual_vs_predicted.png')

    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,dataset,title in zip(axes,['melidos_holdout','korea_last7'],['MeLiDos 별도 평가','국내 마지막 7일']):
        sub=latency[latency.dataset==dataset]
        for i,role in enumerate(['baseline','selected']):
            g=sub[sub.role==role].set_index('latency_band').reindex(['0-10','10-30','30-60','60+'])
            ax.bar(np.arange(4)+(i-.5)*.36,g.mae,width=.36,color=colors[role],label='기준선' if role=='baseline' else 'MLP')
        counts=sub[sub.role=='selected'].set_index('latency_band').reindex(['0-10','10-30','30-60','60+']).n
        ax.set_xticks(range(4),[f'{band}\n(n={int(n)})' for band,n in zip(['0-10','10-30','30-60','60+'],counts)])
        ax.set(title=title,xlabel='실제 입면 지연 구간 (분)',ylabel='구간 내 기록별 MAE (분)')
        ax.legend(frameon=False)
    save(fig,'02_A_error_by_actual_latency.png')

    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    q=quality[(quality.dataset=='melidos_holdout') & (quality.role=='selected')].sort_values('actual')
    axes[0].plot([1,5],[1,5],ls='--',color='#777777',label='실제 점수와 일치')
    axes[0].plot(q.actual,q.predicted_mean,'o-',color=colors['selected'],label='평균 예측 점수')
    axes[0].set(xlabel='실제 수면의 질 (점)',ylabel='평균 예측 (점)',xticks=range(1,6),ylim=(.8,5.2),title='점수 양끝의 예측 편향')
    axes[0].legend(frameon=False)
    for i,role in enumerate(['baseline','selected']):
        g=quality[(quality.dataset=='melidos_holdout') & (quality.role==role)].sort_values('actual')
        axes[1].bar(g.actual+(i-.5)*.34,g.mae,width=.34,color=colors[role],label='기준선' if role=='baseline' else 'MLP')
    axes[1].set_xticks(range(1,6),[f'{int(v)}\n(n={int(n)})' for v,n in zip(q.actual,q.n)])
    axes[1].set(xlabel='실제 수면의 질 (점)',ylabel='점수별 기록 MAE (점)',title='점수별 예측 오차')
    axes[1].legend(frameon=False)
    save(fig,'03_B_error_by_quality.png')

    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    configs=[('A','melidos_holdout','MeLiDos A · 과거 기록 수'),('B','melidos_holdout','MeLiDos B · 과거 기록 수')]
    for ax,(task,dataset,title) in zip(axes,configs):
        for i,role in enumerate(['baseline','selected']):
            g=history[(history.task==task)&(history.dataset==dataset)&(history.role==role)].set_index('history_band').reindex(['0 records','1-2 records','3 records'])
            ax.bar(np.arange(3)+(i-.5)*.36,g.mae,width=.36,color=colors[role],label='기준선' if role=='baseline' else 'MLP')
        ax.set_xticks(range(3),['0건','1~2건','3건'])
        ax.set(title=title,xlabel='입면 지연 과거 유효 기록 수',ylabel=f'기록별 MAE ({"분" if task=="A" else "점"})')
        ax.legend(frameon=False)
    save(fig,'04_error_by_history.png')

    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,task,unit in zip(axes,['A','B'],['분','점']):
        g=sites[(sites.task==task)&(sites.dataset=='melidos_holdout')&(sites.role=='selected')].sort_values('macro_mae')
        ax.barh(g.site_code,g.macro_mae,color=colors['selected'])
        ax.set(xlabel=f'참가자 평균 MAE ({unit})',title=f'{task} · 지역별 오차 (소규모 탐색 분석)')
        for j,r in enumerate(g.itertuples()): ax.text(r.macro_mae+.01,j,f'{r.people}명',va='center',fontsize=9)
        ax.margins(x=.2)
    save(fig,'05_site_errors.png')

    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for ax,task,unit in zip(axes,['A','B'],['분','점']):
        g=ci[(ci.task==task)&(ci.comparison=='baseline')].copy()
        labels={'melidos_holdout':'MeLiDos 37명','korea_last7':'국내 마지막 7일','korea_all':'국내 전체'}
        y=np.arange(len(g))
        ax.errorbar(g.delta_macro_mae,y,xerr=[g.delta_macro_mae-g.lower_95,g.upper_95-g.delta_macro_mae],fmt='o',capsize=5,color=colors['selected'])
        ax.axvline(0,ls='--',color='#777777')
        ax.set_yticks(y,[labels[v] for v in g.dataset])
        ax.set(xlabel=f'MLP - 기준선 MAE 차이 ({unit})',title=f'{task} · 참가자 재표집 95% 구간')
    save(fig,'06_paired_uncertainty.png')

    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    latency_max=max(f.bed_to_sleep_min.max() for f in frames.values())
    duration_max=max(f.sleep_interval_min.max() for f in frames.values())
    latency_edges=np.arange(0,np.ceil(latency_max/20)*20+1,20)
    duration_edges=np.arange(0,np.ceil(duration_max/30)*30+1,30)
    for split,label,color in [('train_melidos_all','MeLiDos 학습','#183A52'),('test_melidos','MeLiDos 평가','#138C8F'),('test_korea','국내 외부','#D88432')]:
        f=frames[split]
        axes[0].hist(f.bed_to_sleep_min.dropna(),bins=latency_edges,alpha=.45,density=True,color=color,label=label)
        axes[1].hist(f.sleep_interval_min.dropna(),bins=duration_edges,alpha=.45,density=True,color=color,label=label)
    axes[0].set(xlabel='입면 지연 (분)',ylabel='확률 밀도',title='입면 지연 분포')
    axes[1].set(xlabel='수면 구간 길이 (분)',ylabel='확률 밀도',title='수면 구간 분포')
    axes[0].legend(frameon=False,fontsize=9)
    save(fig,'07_input_target_distributions.png')

    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,task,unit in zip(axes,['A','B'],['분','점']):
        g=all_metrics[(all_metrics.source=='primary')&(all_metrics.task==task)&(all_metrics.role.isin(['baseline','selected']))]
        dataset_order=['melidos_holdout','korea_last7'] if task=='A' else ['melidos_holdout']
        for i,role in enumerate(['baseline','selected']):
            values=g[g.role==role].set_index('dataset').reindex(dataset_order).macro_mae
            bars=ax.bar(np.arange(len(values))+(i-.5)*.32,values,width=.32,color=colors[role],label='기준선' if role=='baseline' else 'MLP')
            ax.bar_label(bars,fmt='%.2f' if task=='A' else '%.3f',padding=4,fontsize=10)
        ax.set_xticks(range(len(dataset_order)),['MeLiDos 37명','국내 마지막 7일'] if task=='A' else ['MeLiDos 37명'])
        ax.set(ylabel=f'참가자 평균 MAE ({unit})',title=f'{task} · 고정 모델 최종 평가')
        ax.set_ylim(0,g.macro_mae.max()*1.22)
        ax.legend(frameon=False)
    save(fig,'08_primary_test_comparison.png')

    # Typical profile, chosen from TRAIN only. This is model behavior, not efficacy.
    train_b=frames['train_melidos_all'].dropna(subset=['sleep_interval_min','sleep_quality'])
    profile=train_b[FEATURES].median().to_dict()
    lo,hi=train_b.sleep_interval_min.quantile([.05,.95])
    candidates=np.arange(np.ceil(lo/30)*30,np.floor(hi/30)*30+1,30)
    curve=pd.DataFrame([{**profile,'sleep_interval_min':v} for v in candidates])
    curve['predicted_quality']=numpy_forward(saved_models[('B','selected')],curve)
    curve.to_csv(out/'B_duration_curve_train_median_profile.csv',index=False,encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    ax.plot(curve.sleep_interval_min/60,curve.predicted_quality,'o-',color=colors['selected'])
    ax.set(xlabel='후보 수면 구간 길이 (시간)',ylabel='B 예측 수면의 질 (점)',ylim=(1,5),
           title='학습 중앙값 이력에서 수면 구간 길이별 예측')
    ax.text(.02,.02,'학습 자료 5~95백분위 내 30분 간격 · 추천 효과를 검증한 결과 아님',transform=ax.transAxes,fontsize=9)
    save(fig,'09_B_duration_behavior.png')

    hashes={p.relative_to(REPO).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in (DATA_ROOT, code_dir, MODEL_DIR, CV_DIR, TEST_DIR, TREE_DIR)
            for p in folder.rglob('*') if p.is_file() and p.suffix in {'.csv','.py','.joblib','.json'}}
    report={'verification_method':'Raw-source reconstruction + saved CSV metrics + sklearn scaler / NumPy MLP forward pass',
            'no_retraining':True,'original_pytorch_verify_executed':False,
            'full_original_source_verified':True,
            'full_original_source_archive':str(full_team_zip.resolve()) if full_team_zip is not None else None,
            'verification_warnings':verification_warnings,
            'pytorch_note':'PyTorch is not used by this analysis. NumPy forward reconstructs supplied weights and activation functions; original training was not rerun.',
            'checks':checks,'model_verification':inference,'metrics':all_metrics.fillna('').to_dict(orient='records'),
            'paired_intervals':ci.to_dict(orient='records'),
            'split_summary':split_summary,'followup_summary':followup.groupby('split').span_days.agg(['min','median','max']).to_dict(orient='index'),
            'verification_versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                                     'sklearn':sklearn.__version__,'joblib':joblib.__version__,'matplotlib':matplotlib.__version__,
                                     'scipy':scipy.__version__},
            'source_versions':json.loads((ENV_DIR/'versions.json').read_text()),
            'limits':['Prior human use of the 37-person holdout cannot be established from these files.',
                      'Tree holdout comparisons are post hoc reuse of existing evaluation data.',
                      'No external Korea labels for B.',
                      'Observed-duration quality prediction does not establish causal recommendation benefit.'],
            'source_sha256':hashes}
    (out/'audit_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (out/'verification_checks.csv').write_text(pd.DataFrame(checks).to_csv(index=False),encoding='utf-8-sig')
    summary=['# 모델 검증 및 오차 분석 결과','',
             '## 검증 범위',
             '원본 선택 레코드부터 전처리 데이터를 다시 구성하고, 저장된 예측값에서 평가 지표와 참가자 단위 10,000회 재표집 구간을 재계산했다. 저장된 MLP의 결측 대체·표준화와 가중치를 불러와 NumPy로 순전파를 독립 구현했다. 학습이나 Test를 이용한 설정 변경은 하지 않았다.',
             '',f'검증 체크 {len(checks)}개 통과. 저장 모델 비교 {len(inference)}개 통과. 최대 예측 차이 {max(x["max_absolute_prediction_difference"] for x in inference):.8f}.',
             '저장소의 원본 CSV를 직접 읽어 최신 전처리 데이터와 대조했다.',
             '원래 PyTorch verify.py 및 모델 학습 전체는 이 환경에서 실행하지 않았다. NumPy 검증과 원래 학습 재현을 구분해야 한다.',
             '', '## 데이터 관찰 기간',followup.groupby('split')[['records','span_days']].agg(['min','median','max']).to_string(),
             '', '## 주요 최종 성능',all_metrics[(all_metrics.source=='primary') & all_metrics.role.isin(['selected','baseline'])][['task','dataset','role','n','people','macro_mae','rmse','bias_pred_minus_actual','within_tolerance']].to_string(index=False),
             '', '## 실제 입면 지연 구간별 분석',latency.to_string(index=False),
             '', '구간은 오른쪽 끝을 포함한다: [0,10], (10,30], (30,60], (60,무한대). 그룹 그래프는 기록별 MAE이며 주 성능표는 참가자 평균 MAE다.',
             '', '## 실제 수면의 질별 분석',quality.to_string(index=False),
             '', '## 해석에 필요한 제한',
             '- 37명 자료를 팀이 과거 모델 선택에 사용했는지는 코드만으로 확인할 수 없다. 확인 전에는 “별도 참가자 평가”로 표기한다.',
             '- 국내 마지막 7일은 이전까지의 실제 수면일지를 매일 제공받는 조건이다. 앞 21일이 학습용 Train이라는 뜻은 아니다.',
             '- A 국내 MAE와 ±10분 비율에서는 개인 과거 중앙값 기준선이 우세하다. RMSE만 낮아진 것으로 전체 개선을 주장하지 않는다.',
             '- B 최종 입력 제외 비교는 알고리즘도 다르다. 수면 구간 길이 변수만의 비교는 b_same_mlp_ablation_cv.csv를 사용한다.',
             '- 지역 및 작은 점수별 그룹은 소규모 탐색 분석이다. 지역 원인, 인과관계, 추천 치료 효과를 확정하지 않는다.',
             '- 기존 계획의 early stopping은 실제 mlp.py에 구현되어 있지 않다. 실제로는 60/160 epoch를 교차검증으로 선택했다.',
             '- 계획한 A 수면 길이 이력 제외 실험은 동봉 CV 결과에 없다. 완료한 실험으로 표시하지 않는다.',
             '- 관찰된 수면 구간으로 B 점수를 예측한 결과는 후보 추천 효과나 개인 최적 수면 시간을 입증하지 않는다.',
             '', '## 파일',
             '- verified_metrics.csv: 저장된 모든 주 평가와 사후 트리 비교 지표 재검산',
             '- verified_paired_intervals.csv: MLP 대비 기준선 및 입력 제외 모델의 참가자 단위 불확실성',
             '- latency_bands.csv / quality_bands.csv / history_bands.csv / site_errors.csv: 하위집단별 오차',
             '- audit_report.json: 검증 결과, 버전, 원본 SHA-256',
             f'- `../../{FIGURE_DIR.name}`: 완전한 평가 행을 사용한 PNG 그래프 9개',
             f'- `../../{code_dir.name}/analyze_results.py`: 본 분석의 재실행 코드',
             '', f'모든 분석의 자료 출처는 `{DATA_DIR.relative_to(REPO).as_posix()}`, '
             f'`{TEST_DIR.relative_to(REPO).as_posix()}`, `{TREE_DIR.relative_to(REPO).as_posix()}`, '
             f'`{DATA_ROOT.name}/원본`이다.']
    (out/'audit_summary.md').write_text('\n'.join(summary),encoding='utf-8')
    print(json.dumps({'checks':len(checks),'model_comparisons':len(inference),'output':str(out),
                      'max_prediction_difference':max(x['max_absolute_prediction_difference'] for x in inference)},ensure_ascii=False))


if __name__ == '__main__':
    main()
