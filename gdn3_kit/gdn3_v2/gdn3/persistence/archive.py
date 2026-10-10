"""Durable exact original spans. Positions use arbitrary-size decimal integers.

SQLite stores originals and immutable versions; approximate working KV never
replaces this payload. The portable semantic index is an exact streaming disk
scan: bounded RAM, O(history) search time, explicitly not an ANN performance claim.
"""
from pathlib import Path
import hashlib
import heapq
import json
import sqlite3
import time
import numpy as np

class Archive:
    def __init__(self,path,dimension,tokenizer="token-ids-v1"):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        self.path=path;self.dimension=int(dimension);self.tokenizer=tokenizer
        self.db=sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL); CREATE TABLE IF NOT EXISTS records(id INTEGER PRIMARY KEY,seq TEXT NOT NULL,start TEXT NOT NULL,end TEXT NOT NULL,fact_key TEXT,version INTEGER NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL,vector BLOB NOT NULL,tombstone INTEGER NOT NULL DEFAULT 0,UNIQUE(seq,start,end,sha256)); CREATE INDEX IF NOT EXISTS record_key ON records(seq,fact_key,id);")
        expected=json.dumps(dict(schema=1,dimension=self.dimension,tokenizer=tokenizer),sort_keys=True)
        row=self.db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
        if row and row[0]!=expected:raise ValueError("archive schema/dimension/tokenizer mismatch")
        self.db.execute("INSERT OR IGNORE INTO meta VALUES('schema',?)",(expected,));self.db.commit()
    def close(self):self.db.close()
    def append(self,seq,start,end,payload,vector,fact_key=None):
        if int(start)<0 or int(end)<int(start):raise ValueError("invalid causal span")
        vector=np.asarray(vector,dtype=np.float32)
        if vector.shape!=(self.dimension,) or not np.isfinite(vector).all():raise ValueError("invalid index vector")
        raw=json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
        sha=hashlib.sha256(raw.encode()).hexdigest()
        with self.db:
            same=self.db.execute("SELECT id FROM records WHERE seq=? AND start=? AND end=? AND sha256=?",(seq,str(start),str(end),sha)).fetchone()
            if same:return int(same[0])
            version=self.db.execute("SELECT COALESCE(MAX(version),0)+1 FROM records WHERE seq=? AND fact_key IS ?",(seq,fact_key)).fetchone()[0]
            cur=self.db.execute("INSERT INTO records(seq,start,end,fact_key,version,payload,sha256,vector) VALUES(?,?,?,?,?,?,?,?)",(seq,str(start),str(end),fact_key,version,raw,sha,vector.tobytes()))
            return int(cur.lastrowid)
    def _record(self,row):
        rid,seq,start,end,key,version,raw,sha,blob,tomb=row
        if hashlib.sha256(raw.encode()).hexdigest()!=sha:raise ValueError("original payload checksum mismatch")
        return dict(id=rid,seq=seq,start=int(start),end=int(end),key=key,version=version,payload=json.loads(raw),sha256=sha,vector=np.frombuffer(blob,dtype=np.float32).copy(),tombstone=bool(tomb))
    def get(self,rid,before=None):
        row=self.db.execute("SELECT * FROM records WHERE id=?",(int(rid),)).fetchone()
        if row is None:return None
        record=self._record(row)
        if before is not None and record["end"]>=int(before):return None
        return record
    def latest(self,seq,key,before):
        # Decimal positions are filtered as Python integers, never lexically or through SQLite int64.
        rows=self.db.execute("SELECT * FROM records WHERE seq=? AND fact_key=? ORDER BY id DESC",(seq,key))
        for row in rows:
            r=self._record(row)
            if r["end"]<int(before):return None if r["tombstone"] else r
        return None
    def search(self,seq,query,before,limit=16):
        query=np.asarray(query,np.float32)
        if query.shape!=(self.dimension,) or limit<1:raise ValueError("invalid search request")
        norm=max(float(np.linalg.norm(query)),1e-12);heap=[];visited=0;started=time.perf_counter()
        visible={}
        for row in self.db.execute("SELECT id,fact_key,end,tombstone FROM records WHERE seq=? AND fact_key IS NOT NULL ORDER BY id DESC",(seq,)):
            if int(row[2])<int(before) and row[1] not in visible:visible[row[1]]=(int(row[0]),bool(row[3]))
        for row in self.db.execute("SELECT * FROM records WHERE seq=? ORDER BY id",(seq,)):
            if int(row[3])>=int(before) or row[9]:continue
            if row[4] is not None and (row[4] not in visible or visible[row[4]][1] or visible[row[4]][0]!=int(row[0])):continue
            visited+=1;v=np.frombuffer(row[8],np.float32)
            score=float(query@v)/(norm*max(float(np.linalg.norm(v)),1e-12))
            entry=(score,-int(row[0]),int(row[0]))
            if len(heap)<limit:heapq.heappush(heap,entry)
            elif entry>heap[0]:heapq.heapreplace(heap,entry)
        records=[self.get(rid,before) for _,_,rid in sorted(heap,reverse=True)]
        return records,dict(visited=visited,latency_seconds=time.perf_counter()-started,search="exact_disk_scan",disk_bytes=self.disk_bytes())
    def tombstone(self,seq,key,position):
        rid=self.append(seq,position,position,{"deleted_key":key},np.zeros(self.dimension),key)
        with self.db:self.db.execute("UPDATE records SET tombstone=1 WHERE id=?",(rid,))
        return rid
    def cursor(self):return int(self.db.execute("SELECT COALESCE(MAX(id),0) FROM records").fetchone()[0])
    def disk_bytes(self):return sum(p.stat().st_size for p in self.path.parent.glob(self.path.name+"*") if p.is_file())
