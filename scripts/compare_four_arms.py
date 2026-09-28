#!/usr/bin/env python3
"""Compare paired source/compressed Resurface results on identical inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

OLD_MK_SHA='306ae9e8ed5e78756f7c8ea39c8db40dbebf3895ced3c5c279722be5100a344b'
OLD_PPL_SHA='06a71c11fc0a12a52add6e7bf5d28b8a8eb9f832b2ec1cb846d9acaf30afe961'
IDENTITY=('id','condition','prompt_token_sha256_int64le')


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):
            digest.update(chunk)
    return digest.hexdigest()


def identity(rows):
    return [tuple(row[k] for k in IDENTITY) for row in rows]


def summary(rows):
    result={}
    for condition in ('normal','target_removed'):
        relevant=[r for r in rows if r['condition']==condition]
        result[condition]={'correct':sum(r['correct'] for r in relevant),
                           'count':len(relevant)}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-report',type=Path,required=True)
    p.add_argument('--compressed-mk-report',type=Path,required=True)
    p.add_argument('--compressed-ppl-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if (sha256_file(a.compressed_mk_report)!=OLD_MK_SHA or
            sha256_file(a.compressed_ppl_report)!=OLD_PPL_SHA):
        raise ValueError('Historical compressed reports differ from published receipts')
    source=json.loads(a.source_report.read_text())
    mk=json.loads(a.compressed_mk_report.read_text())
    ppl=json.loads(a.compressed_ppl_report.read_text())
    if (source.get('complete') is not True or source.get('smoke') is not False
            or source.get('split')!='confirm' or mk.get('complete') is not True
            or ppl.get('complete') is not True):
        raise ValueError('Only complete full source/COMPRESSED reports are comparable')
    rows={
        'source_fp16':source['baseline_mk']['rows'],
        'source_resurface':source['adapter_mk']['rows'],
        'compressed_base':mk['arms']['current_readapted']['rows'],
        'compressed_resurface':mk['arms']['active_resurface']['rows'],
    }
    if (any(len(item)!=768 for item in rows.values()) or
            len({tuple(identity(item)) for item in rows.values()})!=1):
        raise ValueError('Different MK prompt ordering, tokens or split')
    for name,item in rows.items():
        if summary(item)['normal']['count']!=384 or summary(item)['target_removed']['count']!=384:
            raise ValueError(f'Incomplete MK condition coverage: {name}')
    source_ppl_windows=source['baseline_ppl']['windows']
    source_active_windows=source['adapter_ppl']['windows']
    compressed_ppl_windows=ppl['ppl']['current_readapted']['windows']
    compressed_active_windows=ppl['ppl']['active_resurface']['windows']
    def windows_id(items):
        return [(r['start'],r['target_tokens'],r['token_sha256_int64le']) for r in items]
    if (any(len(item)!=130 for item in (source_ppl_windows,source_active_windows,
                                         compressed_ppl_windows,compressed_active_windows)) or
            len({tuple(windows_id(item)) for item in (source_ppl_windows,source_active_windows,
                compressed_ppl_windows,compressed_active_windows)})!=1 or
            sum(r['target_tokens'] for r in source_ppl_windows)!=264764):
        raise ValueError('PPL tokenizer/window/target coverage differs')
    metrics={name:{'mk':summary(item)} for name,item in rows.items()}
    for name,value in zip(metrics,(
        source['baseline_ppl']['ppl'],source['adapter_ppl']['ppl'],
        ppl['ppl']['current_readapted']['summary']['ppl'],
        ppl['ppl']['active_resurface']['summary']['ppl'])):
        metrics[name]['ppl']=value
    score=lambda name:100*metrics[name]['mk']['normal']['correct']/384
    source_gain=score('source_resurface')-score('source_fp16')
    compressed_gain=score('compressed_resurface')-score('compressed_base')
    result={'format':'MAMBA2_SOURCE_COMPRESSED_FOUR_ARM_COMPARISON_V1',
        'complete':True,'source_report_sha256':sha256_file(a.source_report),
        'compressed_mk_report_sha256':OLD_MK_SHA,'compressed_ppl_report_sha256':OLD_PPL_SHA,
        'exact_mk_prompt_identity':True,'exact_ppl_window_identity':True,
        'metrics':metrics,'source_adapter_gain_pp':source_gain,
        'compressed_adapter_gain_pp':compressed_gain,
        'difference_of_gains_pp':compressed_gain-source_gain,
        'interpretation':'Descriptive four-arm comparison on already observed numeric/template and validation data; the compressed base also received 448 small-tensor readaptation updates before adapter training, and each arm has its own unadapted-base KL teacher. Do not infer a causal quantization-only interaction or unseen-template generalization.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'complete':True,'source_adapter_gain_pp':source_gain,
        'compressed_adapter_gain_pp':compressed_gain,
        'difference_of_gains_pp':compressed_gain-source_gain}),flush=True)


if __name__=='__main__':main()
