"""Host-side persistent retrieval coordinator for model decode.

Archive originals are appended separately by the application at committed span
boundaries. This adapter performs a read-only causal search, projects retrieved
records into model head space, supplies a padded external branch, then appends
new KV vectors to an Oja working cache. It never writes model-visible results
back into the source archive."""
from __future__ import annotations
import numpy as np
from ..compression.oja_kv import OjaKV

class PersistentSession:
    def __init__(self,archive,*,sequence_id,key_dim,value_dim,capacity=128,rank=8,anchors=8,candidates=16,seed=0,head_count=1):
        if candidates<1:raise ValueError("candidate count must be positive")
        self.archive=archive;self.sequence_id=str(sequence_id);self.key_dim=key_dim;self.value_dim=value_dim;self.candidates=candidates;self.head_count=int(head_count)
        self.kv=OjaKV(key_dim,value_dim,capacity,rank,anchors,seed=seed);self.cursor=archive.cursor();self.retrieval_log=[]
    def retrieve(self,query,position,project):
        """project(record)->(key,value); only end < position records are exposed."""
        q=np.asarray(query,np.float32)
        if q.shape!=(self.archive.dimension,):raise ValueError("query must match archive index dimension")
        records,metrics=self.archive.search(self.sequence_id,q,before=int(position),limit=self.candidates)
        vectors=[];values=[];ids=[]
        for r in records:
            result=project(r)
            if len(result)!=2:raise ValueError("projection must return key,value")
            k,v=map(lambda x:np.asarray(x,np.float32),result)
            if k.shape!=(self.head_count,self.key_dim) or v.shape!=(self.head_count,self.value_dim):raise ValueError("projection must return (Ha,Dk) and (Ha,Dv)")
            vectors.append(k);values.append(v);ids.append(r["id"])
        if vectors:
            padded_k=np.zeros((self.head_count,self.candidates,self.key_dim),np.float32);padded_v=np.zeros((self.head_count,self.candidates,self.value_dim),np.float32);valid=np.zeros((self.head_count,self.candidates),bool)
            padded_k[:,:len(vectors)]=np.stack(vectors,axis=1);padded_v[:,:len(values)]=np.stack(values,axis=1);valid[:,:len(ids)]=True
        else:
            padded_k=np.zeros((self.head_count,self.candidates,self.key_dim),np.float32);padded_v=np.zeros((self.head_count,self.candidates,self.value_dim),np.float32);valid=np.zeros((self.head_count,self.candidates),bool)
        self.retrieval_log.append(dict(position=int(position),ids=ids,metrics=metrics))
        return dict(k=padded_k,v=padded_v,valid=valid),metrics
    def append_working(self,key_id,key,value,priority=0.):
        self.kv.put(key_id,key,value,priority);self.cursor=max(self.cursor,self.archive.cursor())
    def state_dict(self):return dict(sequence_id=self.sequence_id,cursor=self.cursor,head_count=self.head_count,kv=self.kv.state_dict(),retrieval_log=list(self.retrieval_log))
    def load_state(self,state):
        if state["sequence_id"]!=self.sequence_id:raise ValueError("persistent sequence identity mismatch")
        if int(state["head_count"])!=self.head_count:raise ValueError("persistent head-count schema mismatch")
        self.cursor=int(state["cursor"]);self.kv=OjaKV.from_state(state["kv"]);self.retrieval_log=list(state["retrieval_log"])
        if self.cursor>self.archive.cursor():raise ValueError("checkpoint cursor is ahead of durable archive")
        return self
