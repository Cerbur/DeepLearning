"""Read-only corpus audit; writes only audit artifacts beside this script. Stdlib only."""
import collections, hashlib, importlib.util, json, math, random, re, unicodedata
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
rng = random.Random(20261020)
def norm(s):
    return ''.join(c for c in unicodedata.normalize('NFKC', s).casefold() if c.isalnum())
def quant(xs):
    a=sorted(xs)
    return {'min':a[0], 'p10':a[round((len(a)-1)*.1)], 'p50':a[round((len(a)-1)*.5)], 'p90':a[round((len(a)-1)*.9)], 'p95':a[round((len(a)-1)*.95)], 'p99':a[round((len(a)-1)*.99)], 'max':a[-1], 'mean':round(sum(a)/len(a),3)}
flags={
 'explicit_abuse_keyword':re.compile(r'傻[逼比B]|煞笔|傻叉|你妈[逼比]|他妈的|妈的|草泥马|操你|艹你|滚蛋|贱人|婊子|狗日|王八蛋|废物|脑残|人渣|屌丝|尼玛|卧槽|我操|我艹'),
 'insult_context_review_only':re.compile(r'废人|厕纸|死胖子|管你屁事|垃圾|眼瞎|智障|弱智|死肥|不要脸|神经病'),
 'sexual_keyword_review_only':re.compile(r'做爱|约炮|性交|强奸|嫖娼|鸡巴|阴茎|阴道|乳房|避孕套|色情|黄片|卖淫|口交'),
 'ad_keyword_review_only':re.compile(r'代购|加[微Vv]|微信号|QQ群|淘宝店|优惠券|包邮|刷单|兼职|招聘|代理商|联系电话'),
 'url':re.compile(r'https?://|www\.'),
 'email':re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
 'mainland_mobile_candidate':re.compile(r'(?<![0-9])1[3-9][0-9]{9}(?![0-9])'),
 'html_candidate':re.compile(r'</?[A-Za-z][^>]*>'),
 'replacement_char':re.compile('\ufffd'),
 'control':re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]'),
 'repeat_char_6plus':re.compile(r'(.)\1{5,}'),
 'repeat_phrase_4plus':re.compile(r'([^\s]{2,8})\1{3,}'),
 'weibo_emoticon':re.compile(r'\[(?:哈哈|嘻嘻|泪|笑cry|允悲|爱你|心|赞|偷笑|挤眼|doge|吃惊|怒|抓狂|嘘|害羞|亲亲|可怜|抱抱|蜡烛|围观|坏笑|鄙视|悲伤|愉快|吐|害羞)\]',re.I),
}
spec=importlib.util.spec_from_file_location('audit_cleaning',ROOT/'corpus/cleaning.py')
cleaning=importlib.util.module_from_spec(spec); spec.loader.exec_module(cleaning)
allrows=[]; report={'method':{'seed':20261020,'normalization':'NFKC+casefold+keep isalnum; diagnostic only, never edits corpus','near_duplicate':'all-pairs prefix join on unique normalized docs >=30 chars; character 4-gram set Jaccard >=0.8','keyword_flags':'heuristic review queues; NOT validated defect rates','manual_sample':'6 uniform rows per train file, fixed seed, 60 total'},'splits':{},'files':[]}
samples=[]; flag_examples=collections.defaultdict(list)
for split in ['train','eval']:
    rows=[]; malformed=[]; chars=collections.Counter(); changed=0
    paths=sorted((ROOT/'.data/corpus'/split).glob('*.jsonl'))
    for p in paths:
        current=[]
        with p.open(encoding='utf-8',errors='strict') as f:
            for line_no,line in enumerate(f,1):
                try:
                    obj=json.loads(line); text=obj['text']; assert isinstance(text,str)
                except (ValueError,KeyError,TypeError,AssertionError) as e:
                    malformed.append({'file':p.relative_to(ROOT).as_posix(),'line':line_no,'error':str(e)}); continue
                row={'split':split,'file':p.relative_to(ROOT).as_posix(),'line':line_no,'text':text}
                current.append(row); rows.append(row); chars.update(text)
                changed += cleaning.clean_corpus_text(text)!=text
        report['files'].append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'rows':len(current),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        if split=='train': samples.extend(rng.sample(current,min(6,len(current))))
    lens=[len(x['text']) for x in rows]; turns=[x['text'].splitlines() for x in rows]; turnlens=[len(t) for ts in turns for t in ts]
    countflags=collections.Counter({name:0 for name in flags})
    for row in rows:
        for name,pat in flags.items():
            if pat.search(row['text']):
                countflags[name]+=1
                if len(flag_examples[split+':'+name])<5: flag_examples[split+':'+name].append(row)
    rawcount=collections.Counter(x['text'] for x in rows); normcount=collections.Counter(norm(x['text']) for x in rows)
    tc=collections.Counter(t.strip() for ts in turns for t in ts if t.strip())
    total=sum(chars.values()); cjk=sum(v for c,v in chars.items() if '\u4e00'<=c<='\u9fff')
    report['splits'][split]={'documents':len(rows),'malformed':malformed,'empty_docs':sum(not x['text'].strip() for x in rows),'bytes':sum(p.stat().st_size for p in paths),'chars':total,'char_lengths':quant(lens),'turn_count':quant([len(ts) for ts in turns]),'turn_lengths':quant(turnlens),'turns_total':len(turnlens),'turns_le_5_chars':sum(n<=5 for n in turnlens),'docs_lt_30_chars':sum(n<30 for n in lens),'docs_ge_256_chars':sum(n>=256 for n in lens),'docs_ge_512_chars':sum(n>=512 for n in lens),'cjk_char_fraction':cjk/total,'ascii_alpha_fraction':sum(v for c,v in chars.items() if c.isascii() and c.isalpha())/total,'newline_fraction':chars['\n']/total,'unique_chars':len(chars),'raw_duplicate_extra_docs':sum(v-1 for v in rawcount.values()),'normalized_duplicate_extra_docs':sum(v-1 for v in normcount.values()),'changed_by_current_cleaner':changed,'flags':dict(countflags),'most_common_turns':tc.most_common(20)}
    allrows.extend(rows)
train=[x for x in allrows if x['split']=='train']; ev=[x for x in allrows if x['split']=='eval']
traw={x['text'] for x in train}; tnorm={norm(x['text']) for x in train}
train_turns={norm(t) for x in train for t in x['text'].splitlines() if len(norm(t))>=15}
eval_turns=[norm(t) for x in ev for t in x['text'].splitlines() if len(norm(t))>=15]
report['cross_split']={'raw_overlap_eval_docs':sum(x['text'] in traw for x in ev),'normalized_overlap_eval_docs':sum(norm(x['text']) in tnorm for x in ev),'eval_turns_ge15':len(eval_turns),'eval_turns_ge15_seen_in_train':sum(t in train_turns for t in eval_turns),'eval_docs_with_shared_turn_ge15':sum(any(len(norm(t))>=15 and norm(t) in train_turns for t in x['text'].splitlines()) for x in ev)}
# Exact token count under hypothetical fresh default vocabulary, without importing torch.
cc=collections.Counter(c for x in train for c in x['text'])
learned=set(c for c,v in sorted(cc.items(),key=lambda kv:(-kv[1],kv[0])) if v>=2 and ord(c)>=128)
learned=set(sorted(learned,key=lambda c:(-cc[c],c))[:8192-260])
report['tokenization']={'basis':'fresh default hybrid min_freq=2 max_vocab=8192, calculated from current train only','vocab_size':260+len(learned)}
for split,rs in [('train',train),('eval',ev)]:
    lengths=[sum(1 if ord(c)<128 or c in learned else len(c.encode('utf-8')) for c in r['text'])+2 for r in rs]
    report['tokenization'][split]={'tokens_including_bos_eos':sum(lengths),'tokens_per_doc':quant(lengths),'docs_ge512_tokens':sum(v>=512 for v in lengths),'packed_rows_at_512':math.ceil((sum(lengths)-1)/512)}
# Global rarity ordering permits complete prefix filtering for set-Jaccard threshold .8.
unique={}
for row in allrows:
    s=norm(row['text'])
    if len(s)>=30: unique.setdefault(s,[]).append(row)
sets={s:set(s[i:i+4] for i in range(len(s)-3)) for s in unique}
df=collections.Counter(g for gs in sets.values() for g in gs)
ordered=sorted(sets,key=lambda s:(len(sets[s]),s))
index=collections.defaultdict(list); pairs=[]; candidates=0; affected=set(); cross_eval=set(); train_affected=set()
for i,s in enumerate(ordered):
    gs=sets[s]; size=len(gs); prefix=sorted(gs,key=lambda g:(df[g],g))[:size-math.ceil(.8*size)+1]
    cand=set()
    for g in prefix:
        cand.update(j for j in index[g] if 5*len(sets[ordered[j]])>=4*size)
    for j in cand:
        other=ordered[j]; hs=sets[other]; inter=len(gs & hs); union=size+len(hs)-inter
        candidates+=1
        if 5*inter>=4*union:
            a,b=unique[s],unique[other]; affected.update((s,other))
            for row in a+b:
                if row['split']=='train': train_affected.add((row['file'],row['line']))
            cross=({r['split'] for r in a+b}=={'train','eval'})
            if cross:
                cross_eval.update((r['file'],r['line']) for r in a+b if r['split']=='eval')
            if len(pairs)<30: pairs.append({'jaccard':inter/union,'cross_split':cross,'a':a[0],'b':b[0]})
    for g in prefix: index[g].append(i)
report['near_duplicates']={'eligible_unique_normalized_docs':len(sets),'candidate_pairs_verified':candidates,'affected_unique_normalized_docs':len(affected),'train_affected_docs':len(train_affected),'eval_docs_in_cross_split_pairs':len(cross_eval),'example_pairs':pairs}
report['flag_examples']=dict(flag_examples)
# Redact phone-like strings in audit artifacts; never copy contact numbers into reports.
for examples in report['flag_examples'].values():
    for row in examples: row['text']=flags['mainland_mobile_candidate'].sub('<PHONE_REDACTED>',row['text'])
for pair in pairs:
    for side in ['a','b']: pair[side]['text']=flags['mainland_mobile_candidate'].sub('<PHONE_REDACTED>',pair[side]['text'])
(OUT/'corpus_audit_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 固定随机抽样：每个训练分片 6 条，共 60 条','']
for i,r in enumerate(samples,1):
    lines.extend([f"## {i}. {r['file']}:{r['line']}",r['text'].replace('\n',' / '),''])
(OUT/'corpus_audit_samples.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['flag_examples','near_duplicates','files']},ensure_ascii=False,indent=2))
print('NEAR_DUPLICATES',json.dumps({k:v for k,v in report['near_duplicates'].items() if k!='example_pairs'},ensure_ascii=False))
print('AUDIT_COMPLETE')
