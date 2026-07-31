#!/usr/bin/env python3
"""Faithful port of the official funasr-cli.cpp forward (fbank+LFR + SAN-M
encoder + adaptor + Qwen3-0.6B) run against model.pt, to produce structural
golden references: LFR features [T,560], adaptor output [T,1024], n_aud, and
the reference transcription for a given 16k mono f32 clip.

Outputs (to --out dir): lfr_<tag>.bin (f32 [T,560]), adp_<tag>.bin (f32 [T,1024]),
meta_<tag>.json (T, n_aud, dims), text_<tag>.txt.
"""
import argparse, json, math, os, sys
import numpy as np
import torch

FS=16000; WINLEN=400; SHIFT=160; NFFT=512; NMEL=80; LFR_M=7; LFR_N=6
PREEMPH=0.97; LOWF=20.0; HIGHF=8000.0
def hz2mel(f): return 1127.0*math.log(1.0+f/700.0)

def compute_fbank(wav):
    # port of funasr-cli.cpp compute_fbank
    wav = wav.astype(np.float64)*32768.0
    win = np.array([0.54-0.46*math.cos(2.0*math.pi*i/(WINLEN-1)) for i in range(WINLEN)])
    NBIN=NFFT//2+1; bw=FS/NFFT; ml=hz2mel(LOWF); mh=hz2mel(HIGHF); dm=(mh-ml)/(NMEL+1)
    fb=np.zeros((NMEL,NBIN))
    for m in range(NMEL):
        L=ml+m*dm; C=ml+(m+1)*dm; R=ml+(m+2)*dm
        for k in range(NBIN):
            mf=hz2mel(bw*k)
            if mf>L and mf<R:
                fb[m,k]=(mf-L)/(C-L) if mf<=C else (R-mf)/(R-C)
    N=len(wav); T=(N-WINLEN)//SHIFT+1
    fl=1.1920929e-07
    feat=np.zeros((T,NMEL))
    for t in range(T):
        s=wav[t*SHIFT:t*SHIFT+WINLEN].copy()
        mn=s.mean(); fr=s-mn
        # preemph in-place: for i=WINLEN-1..1: fr[i]-=PRE*fr[i-1]; fr[0]-=PRE*fr[0]
        fr2=fr.copy()
        for i in range(WINLEN-1,0,-1):
            fr2[i]-=PREEMPH*fr[i-1]
        fr2[0]-=PREEMPH*fr[0]
        buf=np.zeros(NFFT); buf[:WINLEN]=fr2*win
        spec=np.fft.rfft(buf, NFFT)
        power=(spec.real**2+spec.imag**2)
        for m in range(NMEL):
            e=np.sum(np.where(fb[m]>0, fb[m]*power, 0.0))
            feat[t,m]=math.log(e if e>fl else fl)
    # LFR
    pad=(LFR_M-1)//2; T_lfr=(T+LFR_N-1)//LFR_N
    pd=[feat[0]]*pad+[feat[t] for t in range(T)]
    while len(pd)<(T_lfr-1)*LFR_N+LFR_M: pd.append(feat[T-1])
    D=LFR_M*NMEL; out=np.zeros((T_lfr,D))
    for i in range(T_lfr):
        for j in range(LFR_M):
            out[i,j*NMEL:(j+1)*NMEL]=pd[i*LFR_N+j]
    return out.astype(np.float32), T_lfr

def add_posenc(x, depth):
    T=x.shape[0]
    inc=math.log(10000.0)/(depth/2.0-1.0)
    for t in range(T):
        pos=t+1
        for i in range(depth//2):
            its=math.exp(i*-inc); st=pos*its
            x[t,i]+=math.sin(st); x[t,depth//2+i]+=math.cos(st)
    return x

def ln(x, w, b, eps=1e-5):
    m=x.mean(-1,keepdim=True); v=x.var(-1,unbiased=False,keepdim=True)
    return (x-m)/torch.sqrt(v+eps)*w+b

def sanm_attn(sd,p,x,D=512,H=4,K=11):
    dk=D//H; T=x.shape[0]
    qkv=x@sd[p+'linear_q_k_v.weight'].T+sd[p+'linear_q_k_v.bias']
    q=qkv[:,:D]; k=qkv[:,D:2*D]; v=qkv[:,2*D:]
    # fsmn: fk stored (D,1,K); symmetric pad K//2 -> wait official pad=(K-1)/2=5
    fk=sd[p+'fsmn_block.weight'][:,0,:]  # (D,K)
    pad=(K-1)//2
    vp=torch.cat([torch.zeros(pad,D),v,torch.zeros(pad,D)],0)
    fsmn=v.clone()
    for j in range(K):
        fsmn=fsmn+vp[j:j+T]*fk[:,j]
    qh=q.view(T,H,dk).permute(1,0,2); kh=k.view(T,H,dk).permute(1,0,2); vh=v.view(T,H,dk).permute(1,0,2)
    scores=(qh@kh.transpose(1,2))/math.sqrt(dk)
    probs=torch.softmax(scores,-1)
    o=(probs@vh).permute(1,0,2).reshape(T,D)
    return (o@sd[p+'linear_out.weight'].T+sd[p+'linear_out.bias'])+fsmn

def sanm_layer(sd,p,x,res):
    r=x; h=ln(x,sd[p+'norm1.weight'],sd[p+'norm1.bias'])
    sa=sanm_attn(sd,p+'self_attn.',h)
    x=r+sa if res else sa; r=x
    h=ln(x,sd[p+'norm2.weight'],sd[p+'norm2.bias'])
    h=torch.relu(h@sd[p+'feed_forward.w_1.weight'].T+sd[p+'feed_forward.w_1.bias'])
    h=h@sd[p+'feed_forward.w_2.weight'].T+sd[p+'feed_forward.w_2.bias']
    return r+h

def adp_layer(sd,p,x,D=1024,H=8):
    dk=D//H; T=x.shape[0]; r=x
    h=ln(x,sd[p+'norm1.weight'],sd[p+'norm1.bias'])
    q=(h@sd[p+'self_attn.linear_q.weight'].T+sd[p+'self_attn.linear_q.bias']).view(T,H,dk).permute(1,0,2)
    k=(h@sd[p+'self_attn.linear_k.weight'].T+sd[p+'self_attn.linear_k.bias']).view(T,H,dk).permute(1,0,2)
    v=(h@sd[p+'self_attn.linear_v.weight'].T+sd[p+'self_attn.linear_v.bias']).view(T,H,dk).permute(1,0,2)
    scores=(q@k.transpose(1,2))/math.sqrt(dk); probs=torch.softmax(scores,-1)
    o=(probs@v).permute(1,0,2).reshape(T,D)
    x=r+(o@sd[p+'self_attn.linear_out.weight'].T+sd[p+'self_attn.linear_out.bias']); r=x
    h=ln(x,sd[p+'norm2.weight'],sd[p+'norm2.bias'])
    h=torch.relu(h@sd[p+'feed_forward.w_1.weight'].T+sd[p+'feed_forward.w_1.bias'])
    h=h@sd[p+'feed_forward.w_2.weight'].T+sd[p+'feed_forward.w_2.bias']
    return r+h

@torch.no_grad()
def run_encoder(sd,lfr):
    x=torch.from_numpy(lfr.copy())
    x=x*math.sqrt(512.0)
    add_posenc(x,560)
    ae='audio_encoder.'
    x=sanm_layer(sd,ae+'encoders0.0.',x,False)
    for i in range(49):
        x=sanm_layer(sd,ae+f'encoders.{i}.',x,True)
    x=ln(x,sd[ae+'after_norm.weight'],sd[ae+'after_norm.bias'])
    for i in range(20):
        x=sanm_layer(sd,ae+f'tp_encoders.{i}.',x,True)
    x=ln(x,sd[ae+'tp_norm.weight'],sd[ae+'tp_norm.bias'])
    ad='audio_adaptor.'
    x=torch.relu(x@sd[ad+'linear1.weight'].T+sd[ad+'linear1.bias'])
    x=x@sd[ad+'linear2.weight'].T+sd[ad+'linear2.bias']
    for i in range(2):
        x=adp_layer(sd,ad+f'blocks.{i}.',x)
    return x

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pcm',required=True); ap.add_argument('--tag',required=True)
    ap.add_argument('--out',required=True); ap.add_argument('--model_pt',required=True)
    ap.add_argument('--qwen_dir',required=True)
    ap.add_argument('--skip_llm',action='store_true')
    args=ap.parse_args()
    os.makedirs(args.out,exist_ok=True)
    torch.set_num_threads(8)
    wav=np.fromfile(args.pcm,dtype=np.float32)
    print(f'[{args.tag}] samples={len(wav)} dur={len(wav)/16000:.2f}s',flush=True)
    lfr,T=compute_fbank(wav)
    print(f'[{args.tag}] LFR frames T={T}',flush=True)
    sd_full=torch.load(args.model_pt,map_location='cpu'); sd_full=sd_full.get('state_dict',sd_full)
    sd={k:v.float() for k,v in sd_full.items() if k.startswith('audio_')}
    adp=run_encoder(sd,lfr)
    ol=1+(T-3+2)//2; ol=1+(ol-3+2)//2; n_aud=(ol-1)//2+1
    print(f'[{args.tag}] adaptor {tuple(adp.shape)} n_aud={n_aud}',flush=True)
    lfr.astype(np.float32).tofile(os.path.join(args.out,f'lfr_{args.tag}.bin'))
    adp.numpy().astype(np.float32).tofile(os.path.join(args.out,f'adp_{args.tag}.bin'))
    meta={'T':int(T),'n_aud':int(n_aud),'feature_dim':560,'llm_dim':1024,'d_model':512}
    text=None
    if not args.skip_llm:
        from transformers import AutoTokenizer, Qwen3ForCausalLM, AutoConfig
        cfg=AutoConfig.from_pretrained(args.qwen_dir)
        model=Qwen3ForCausalLM(cfg).eval()
        llm_sd={k[len('llm.'):]:v.float() for k,v in sd_full.items() if k.startswith('llm.')}
        missing,unexpected=model.load_state_dict(llm_sd,strict=False)
        print(f'[{args.tag}] llm load missing={len(missing)} unexpected={len(unexpected)}',flush=True)
        if missing: print('  missing sample:',missing[:5],flush=True)
        if unexpected: print('  unexpected sample:',unexpected[:5],flush=True)
        tok=AutoTokenizer.from_pretrained(args.qwen_dir)
        prefix='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n语音转写：'
        suffix='<|im_end|>\n<|im_start|>assistant\n'
        pre=tok(prefix,return_tensors='pt',add_special_tokens=False).input_ids
        suf=tok(suffix,return_tensors='pt',add_special_tokens=False).input_ids
        print(f'[{args.tag}] prefix_tokens={pre.shape[1]} suffix_tokens={suf.shape[1]}',flush=True)
        emb=model.get_input_embeddings()
        pre_e=emb(pre)[0]; suf_e=emb(suf)[0]
        aud_e=adp[:n_aud].to(pre_e.dtype)
        inp=torch.cat([pre_e,aud_e,suf_e],0).unsqueeze(0)
        with torch.no_grad():
            out=model.generate(inputs_embeds=inp,max_new_tokens=256,do_sample=False,num_beams=1)
        text=tok.decode(out[0],skip_special_tokens=True)
        print(f'[{args.tag}] TEXT: {text}',flush=True)
        with open(os.path.join(args.out,f'text_{args.tag}.txt'),'w') as f: f.write(text)
        meta['prefix_tokens']=int(pre.shape[1]); meta['suffix_tokens']=int(suf.shape[1])
    json.dump(meta,open(os.path.join(args.out,f'meta_{args.tag}.json'),'w'),indent=2)
    print(f'[{args.tag}] DONE',flush=True)

if __name__=='__main__': main()
