"""Native operator-only invariance probe; random tensors, not model quality.

Bind an idle GPU explicitly before running. Tests causal prefix alignment by
cropping both compressed key levels with the native compression formulas.
"""
import argparse
import json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tokens',type=int,default=16384)
    p.add_argument('--fused',action='store_true')
    args=p.parse_args()
    import torch
    from sglang.srt.layers.attention.minicpm.sparse_utils import (
        compressed_attention,compressed_attention_tilelang)
    n=args.tokens; context=32768; torch.manual_seed(42)
    q=torch.randn(n,32,128,device='cuda',dtype=torch.bfloat16)
    k1=torch.randn((n-32)//16+1,2,128,device='cuda',dtype=torch.bfloat16)
    k2=torch.randn((n-128)//64+1,2,128,device='cuda',dtype=torch.bfloat16)
    kernel=None
    if args.fused:
        from sglang.srt.layers.attention.minicpm.fuse_kernel import fused_attn_pooling_online_topk_prefill
        kernel=fused_attn_pooling_online_topk_prefill(groups=16,heads=32,dim=128,
            topk=128,pooled_k_len=512,m_block_dim=16,block_M=16,block_stride=4,
            pad_len=1,num_offs=5,kernel_stride=16,block_size=64,dense_len=8192,
            init_blocks=1,local_blocks=32,dtype_str='bfloat16',batch_size=1,
            max_seqlen_q_grid=n)
    def tensor(values):return torch.tensor(values,device='cuda',dtype=torch.int32)
    def run(h):
        a=k1[:(h-32)//16+1];b=k2[:(h-128)//64+1]
        cu=tensor([0,h]);ca=tensor([0,a.shape[0]]);cb=tensor([0,b.shape[0]])
        if args.fused:
            return compressed_attention_tilelang(q[:h],a,64,96,128,cu,ca,
                cache_lens=tensor([0]),fused_kernel=kernel,max_cache_len=context)
        return compressed_attention(q[:h],a,b,16,64,96,cu,ca,cb,h,context,
            init_blocks=1,local_blocks=32,cache_lens=tensor([0]),
            cu_seqlens_q_adjusted=cu*16,max_seqlen_q_adjusted=h*16)
    reference=run(n); torch.cuda.synchronize()
    passed=True
    for h in [n,n//8,n//4,n//2,n*3//4]:
        selected=run(h);torch.cuda.synchronize();diff=reference[:,:h]!=selected
        data=dict(kind='random_operator_probe',fused=args.fused,tokens=n,head=h,
            equal=not bool(diff.any()),mismatches=int(diff.sum()),
            sparse_suffix_mismatches=int(diff[:,8192:].sum()))
        passed=passed and data['equal'];print(json.dumps(data),flush=True)
    raise SystemExit(0 if passed else 1)


if __name__=='__main__':main()
