"""Build the user-requested 75k balanced subset entirely from saved images."""
from pathlib import Path
import csv,collections,json,os,random,statistics,zlib
from concurrent.futures import ThreadPoolExecutor

root=Path(__file__).resolve().parent/'datasets/plantnet-300k-v2'
source=root/'balanced-80000'; dest=root/'balanced-75000';dest.mkdir(exist_ok=True)
def read(path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def write(path,rows,fields):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
available=collections.defaultdict(list)
def verify(r):
    path=source/r['relative_path']
    if not path.exists():return None
    content=path.read_bytes()
    assert len(content)==int(r['file_size']) and zlib.crc32(content)==int(r['crc32']),str(path)
    return r
with ThreadPoolExecutor(max_workers=16) as pool:
    for i,r in enumerate(pool.map(verify,read(source/'manifest.csv'))):
        if r is not None:available[int(r['species_id'])].append(r)
        if i%10000==0:print('Checked source entries:',i,flush=True)
print('Verified saved files:',sum(map(len,available.values())),flush=True)
capacities={int(r['species_id']):int(r['available']) for r in read(source/'species_counts.csv')}
quota={s:0 for s in available};rng=random.Random(20260927);order=sorted(quota);rng.shuffle(order);total=0
while total<75000:
    for s in order:
        if quota[s]<len(available[s]):quota[s]+=1;total+=1
        if total==75000:break
selected=[];counts=[]
for s,rows in sorted(available.items()):
    groups={k:[r for r in rows if r['split']==k] for k in ['train','val','test']}
    ideal={k:quota[s]*len(v)/len(rows) for k,v in groups.items()};sq={k:int(v) for k,v in ideal.items()}
    for k in sorted(groups,key=lambda k:(-(ideal[k]-sq[k]),k))[:quota[s]-sum(sq.values())]:sq[k]+=1
    for split,group in groups.items():selected.extend(rng.sample(group,sq[split]))
    counts.append({'species_id':s,'species':rows[0]['species'],'available_in_full_source':capacities[s],'available_locally':len(rows),'selected':quota[s],**sq})
assert len(selected)==len({r['PN_hash'] for r in selected})==75000
def link(r):
    target=dest/r['relative_path'];target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():os.link(source/r['relative_path'],target)
    else:assert os.path.samefile(source/r['relative_path'],target)
with ThreadPoolExecutor(max_workers=12) as pool:
    for i,_ in enumerate(pool.map(link,selected)):
        if i%20000==0:print('Linked images:',i,flush=True)
write(dest/'manifest.csv',selected,list(selected[0]))
for s in ['train','val','test']:write(dest/f'{s}.csv',[r for r in selected if r['split']==s],list(selected[0]))
write(dest/'species_counts.csv',counts,list(counts[0]))
(dest/'class_names.json').write_text(json.dumps({r['species_id']:r['species'] for r in counts},ensure_ascii=False,indent=2),encoding='utf-8')
obs=collections.defaultdict(set)
for r in selected:obs[r['PN_observation_id']].add(r['split'])
assert all(len(v)==1 for v in obs.values())
summary={'status':'complete','target':75000,'verified_images':75000,'species':len(quota),'min_per_species':min(quota.values()),'max_per_species':max(quota.values()),'mean':75,'population_stddev':statistics.pstdev(quota.values()),'split_counts':dict(collections.Counter(r['split'] for r in selected)),'species_per_split':{s:len({r['species_id'] for r in selected if r['split']==s}) for s in ['train','val','test']},'saved_before_stop':sum(map(len,available.values())),'extra_saved_images_not_in_subset':sum(map(len,available.values()))-75000,'species_at_local_capacity':sum(quota[s]==len(available[s]) for s in quota),'unexhausted_quotas':sorted({quota[s] for s in quota if quota[s]<len(available[s])}),'seed':20260927,'method':'Water filling minimizes variance subject to LOCALLY SAVED species capacities; largest-remainder source split proportions; seeded uniform sampling without replacement within saved species/split pools. Initial download used circular blocks.','verification':'All 79,831 saved files checked against source ZIP CRC32 and size; final 75,000 paths are hard links to verified files. Official metadata MD5 verified. No selected observation IDs overlap splits.','storage':'NTFS hard links avoid duplicate image disk space. Editing either linked copy changes both.','source':'https://zenodo.org/records/10419064','image_bytes':sum(int(r['file_size']) for r in selected)}
(dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
old=json.loads((source/'summary.json').read_text());old.update(status='stopped_by_user',saved_images=summary['saved_before_stop'],active_subset='../balanced-75000',note='Do not use the original 80k manifests: 169 requested images were not downloaded. Use balanced-75000.')
(source/'summary.json').write_text(json.dumps(old,indent=2))
progress=json.loads((source/'progress.json').read_text());progress['status']='stopped_by_user';(source/'progress.json').write_text(json.dumps(progress,indent=2))
print(json.dumps(summary,indent=2),flush=True)
