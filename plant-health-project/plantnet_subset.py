"""Reproducible balanced Pl@ntNet subset, fetched using HTTP byte ranges."""
from pathlib import Path
import argparse, collections, csv, hashlib, io, json, random, statistics, struct, time, zipfile, zlib
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent / 'datasets/plantnet-300k-v2'
OUT = ROOT / 'balanced-80000'
URL = 'https://zenodo.org/api/records/10419064/files/images.zip/content'
SIZE = 41819638938
CD_OFFSET = 41773274853
SEED = 20260927
RATE_LOCK=threading.Lock()
LAST_REQUEST=0.0

def fetch(start, end):
    for attempt in range(9):
        try:
            global LAST_REQUEST
            with RATE_LOCK:
                time.sleep(max(0,.65-(time.monotonic()-LAST_REQUEST)))
                LAST_REQUEST=time.monotonic()
            response = requests.get(URL, headers={'Range':f'bytes={start}-{end}'}, timeout=(30,180), stream=True)
            if response.status_code in (429,500,502,503,504):
                response.close(); time.sleep(min(120, 10*(attempt+1))); continue
            response.raise_for_status()
            expected = f'bytes {start}-{end}/{SIZE}'
            if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                response.close(); raise RuntimeError('Server did not return the requested byte range')
            data=response.content
            if len(data)!=end-start+1: raise RuntimeError('Incomplete range')
            return data
        except (requests.RequestException,RuntimeError):
            if attempt==8: raise
            time.sleep(min(60,5*(attempt+1)))
    raise RuntimeError('Range retries exhausted')

def readcsv(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def writecsv(path, rows, fields):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def plan():
    OUT.mkdir(parents=True,exist_ok=True)
    rec=json.loads((ROOT/'zenodo_record.json').read_text())
    for f in rec['files']:
        if f['key']=='images.zip': continue
        digest=hashlib.md5((ROOT/f['key']).read_bytes()).hexdigest()
        assert 'md5:'+digest == f['checksum'], f['key']
    central=ROOT/'zip_central_directory.bin'
    if not central.exists():
        print('Downloading ZIP directory (46 MB)',flush=True)
        central.write_bytes(fetch(CD_OFFSET,SIZE-1))
    archive=zipfile.ZipFile(io.BytesIO(central.read_bytes()))
    info={}
    for z in archive.infolist():
        if z.filename.lower().endswith(('.jpg','.jpeg','.png')):
            info[Path(z.filename).stem]=(z,z.header_offset+CD_OFFSET)
    print('Archive images:',len(info), 'example:',next(iter(info.values()))[0].filename,flush=True)
    species={int(r['species_id']):r for r in readcsv(ROOT/'species_metadata.csv')}
    byspecies=collections.defaultdict(list)
    for r in readcsv(ROOT/'plantnet300K_metadata.csv'):
        assert r['PN_hash'] in info, r['PN_hash']
        byspecies[int(r['species_id'])].append(r)
    # Water filling minimizes sum of squared class counts for fixed total and capacities.
    quota={s:0 for s in byspecies}; rng=random.Random(SEED)
    tieorder=sorted(quota);rng.shuffle(tieorder)
    total=0
    while total<80000:
        for s in tieorder:
            if quota[s]<len(byspecies[s]):quota[s]+=1;total+=1
            if total==80000:break
    chosen=[]; counts=[]
    for s,rows in sorted(byspecies.items()):
        groups={k:[r for r in rows if r['split']==k] for k in ['train','val','test']}
        q=quota[s]
        # Largest remainder allocation follows each species' source split proportions.
        ideal={k:q*len(v)/len(rows) for k,v in groups.items()}
        splitq={k:int(v) for k,v in ideal.items()}
        for k in sorted(groups,key=lambda k:(-(ideal[k]-splitq[k]),k))[:q-sum(splitq.values())]:splitq[k]+=1
        for split, group in groups.items():
            group.sort(key=lambda r:info[r['PN_hash']][1])
            n=splitq[split]
            if not n:continue
            # Random circular consecutive block reduces network transfers. No organ-based filtering.
            start=rng.randrange(len(group))
            selected=[group[(start+i)%len(group)] for i in range(n)]
            for r in selected:
                z,offset=info[r['PN_hash']]
                chosen.append(dict(r, species=species[s]['species'],family=species[s]['family'],
                    relative_path=f"images/{split}/{s:04d}/{Path(z.filename).name}",
                    archive_path=z.filename,offset=offset,compressed_size=z.compress_size,
                    file_size=z.file_size,crc32=z.CRC,compression=z.compress_type))
        counts.append({'species_id':s,'species':species[s]['species'],'available':len(rows),'selected':q,**splitq})
    assert len(chosen)==80000 and len({r['PN_hash'] for r in chosen})==80000
    fields=list(chosen[0])
    writecsv(OUT/'manifest.csv',chosen,fields)
    writecsv(OUT/'species_counts.csv',counts,list(counts[0]))
    for split in ['train','val','test']:writecsv(OUT/f'{split}.csv',[r for r in chosen if r['split']==split],fields)
    # Merge neighboring members; cap request size to 24 MB and unused gaps to 32 KB.
    ordered=sorted(chosen,key=lambda r:r['offset']); batches=[]
    for r in ordered:
        end=r['offset']+30+len(r['archive_path'].encode('utf-8'))+65535+r['compressed_size']-1
        # ZIP local extra can be at most 65535 bytes. A small overlap is harmless.
        end=min(end,CD_OFFSET-1)
        if batches and r['offset']<=batches[-1]['end']+32768 and end-batches[-1]['start']<24*1024*1024:
            batches[-1]['end']=max(end,batches[-1]['end']);batches[-1]['rows'].append(r)
        else:batches.append({'start':r['offset'],'end':end,'rows':[r]})
    (OUT/'download_plan.json').write_text(json.dumps(batches),encoding='utf-8')
    nums=list(quota.values())
    report={'target':80000,'species':len(nums),'min_per_species':min(nums),'max_per_species':max(nums),'mean':statistics.mean(nums),'population_stddev':statistics.pstdev(nums),'seed':SEED,'split_counts':dict(collections.Counter(r['split'] for r in chosen)),'requests':len(batches),'planned_transfer_bytes':sum(b['end']-b['start']+1 for b in batches),'method':'capacity-constrained water filling; proportional original splits; seeded circular archive-order blocks within species and split','source':URL,'status':'planned'}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

def download():
    batches=json.loads((OUT/'download_plan.json').read_text())
    summary=json.loads((OUT/'summary.json').read_text());summary['status']='downloading'
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    progress=OUT/'progress.json';done=0; transferred=0;t0=time.time()
    def work(b):
        needed=[r for r in b['rows'] if not (OUT/r['relative_path']).exists() or (OUT/r['relative_path']).stat().st_size!=r['file_size']]
        received=0
        if needed:
            data=fetch(b['start'],b['end']);received=len(data)
            for r in needed:
                at=r['offset']-b['start'];header=struct.unpack_from('<4s5H3L2H',data,at)
                assert header[0]==b'PK\x03\x04'
                body=at+30+header[-2]+header[-1]
                compressed=data[body:body+r['compressed_size']]
                raw=zlib.decompress(compressed,-15) if r['compression']==8 else compressed
                assert len(raw)==r['file_size'] and zlib.crc32(raw)==r['crc32'],r['archive_path']
                target=OUT/r['relative_path'];target.parent.mkdir(parents=True,exist_ok=True)
                temp=target.with_suffix('.part');temp.write_bytes(raw);temp.replace(target)
        return len(b['rows']),received
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures=[pool.submit(work,b) for b in batches]
        for i,future in enumerate(as_completed(futures)):
            images,received=future.result();done+=images;transferred+=received
            status={'completed_images':done,'total_images':80000,'batch':i+1,'total_batches':len(batches),'transferred_bytes_this_run':transferred,'elapsed_seconds':round(time.time()-t0),'status':'downloading'}
            progress.write_text(json.dumps(status,indent=2))
            if i%25==0:print(json.dumps(status),flush=True)
    # Verify all saved bytes, including resumed files, against ZIP CRCs.
    for b in batches:
        for r in b['rows']:
            raw=(OUT/r['relative_path']).read_bytes()
            assert len(raw)==r['file_size'] and zlib.crc32(raw)==r['crc32']
    summary['status']='complete';summary['verified_images']=done;summary['verification']='Every image checked against original ZIP CRC32 and size; metadata MD5 checked against Zenodo'
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));status['status']='complete';progress.write_text(json.dumps(status,indent=2))
    print('COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','download']);args=parser.parse_args()
    plan() if args.action=='plan' else download()
