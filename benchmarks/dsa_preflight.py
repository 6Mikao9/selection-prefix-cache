"""Read-only DSA backend preflight, never launches a model or CUDA kernel.
By default emits installed source evidence only. --check-device explicitly queries
CUDA capability; run only after independently confirming the selected GPU idle
and healthy. Passing is necessary, not proof of kernel/model compatibility.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--check-device',action='store_true')
    a=p.parse_args()
    target=Path(a.output)
    if target.exists(): p.error('--output must be a new file')
    report={'status':'not_benchmarked','versions':{},'sources':[],
            'limitations':['No model or kernel benchmark is performed.',
                'CUDA DSA ragged scoring requires native DeepGEMM FP8 MQA logits; no torch-score substitute.',
                'Capability >= 9 is necessary; SM120 support depends on the installed native backend.',
                'This preflight does not establish GPU availability or reserve a card.']}
    distributions=importlib.metadata.packages_distributions()
    report['distributions']={}
    for package in ('sglang','torch','deep_gemm'):
        names=distributions.get(package, [])
        versions={}
        for name in names:
            try: versions[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: versions[name]='unknown metadata'
        report['distributions'][package]=versions
        report['versions'][package]=versions if versions else 'unknown metadata'
    # Locate packages without importing their CUDA initialization/build code.
    for package, relative in [('sglang','srt/layers/attention/dsa/dsa_indexer.py'),
                              ('deep_gemm','include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh'),
                              ('deep_gemm','include/deep_gemm/impls/sm100_mqa_logits.cuh'),
                              ('deep_gemm','include/deep_gemm/impls/sm120_fp8_mqa_logits.cuh')]:
        spec=importlib.util.find_spec(package)
        roots=list(spec.submodule_search_locations or []) if spec else []
        source=Path(roots[0])/relative if roots else None
        row={'package':package,'relative_source':relative,'present':bool(source and source.is_file())}
        if row['present']:
            raw=source.read_bytes(); text=raw.decode('utf-8')
            row.update(sha256=hashlib.sha256(raw).hexdigest(),
                evidence=[{'line':i,'text':line.strip()} for i,line in enumerate(text.splitlines(),1)
                          if any(term in line for term in ('deep_gemm.fp8_mqa_logits(',
                              'elif _is_xpu:', 'if _is_hip:', 'cute/arch/cluster_sm90.hpp',
                              'cute/arch/mma_sm90_desc.hpp'))])
        report['sources'].append(row)
    if a.check_device:
        import torch
        if not torch.cuda.is_available():
            report['device_gate']={'passed':False,'reason':'CUDA unavailable'}
        else:
            capability=torch.cuda.get_device_capability()
            report['device_gate']={'passed':capability[0]>=9,'capability':list(capability),
                'name':torch.cuda.get_device_name(),
                'reason':'Capability gate only; backend execution remains unverified'}
    else:report['device_gate']={'checked':False}
    with target.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({'status':report['status'],'device_gate':report['device_gate']}))


if __name__=='__main__':main()
