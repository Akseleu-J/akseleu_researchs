"""Bounded online KV cache: exact priority anchors and versioned Oja bases."""
from collections import OrderedDict
import numpy as np

class OjaKV:
    def __init__(self,key_dim,value_dim,capacity=128,rank=8,anchors=8,eta=.05,seed=0):
        if not 1<=rank<=min(key_dim,value_dim) or not 0<=anchors<=capacity:raise ValueError("invalid Oja dimensions/budget")
        self.kdim,self.vdim,self.capacity,self.rank,self.anchors=key_dim,value_dim,capacity,rank,anchors
        self.eta=float(eta);self.count=0;self.version=0;self.entries=OrderedDict()
        rng=np.random.default_rng(seed)
        self.qk=np.linalg.qr(rng.normal(size=(key_dim,rank)))[0]
        self.qv=np.linalg.qr(rng.normal(size=(value_dim,rank)))[0]
    def _decode(self,e):
        if e["exact"]:return e["k"].copy(),e["v"].copy()
        if e["basis_version"]!=self.version:raise ValueError("stale KV coordinate system")
        return self.qk@e["k"],self.qv@e["v"]
    def _basis(self,q,x):
        a=x@q;rate=self.eta/np.sqrt(max(self.count,1))
        update=np.outer(x,a)-q@np.outer(a,a)
        q,_=np.linalg.qr(q+rate*update)
        return q[:,:self.rank]
    def put(self,key,k,v,priority=0.):
        k,v=np.asarray(k,np.float64),np.asarray(v,np.float64)
        if k.shape!=(self.kdim,) or v.shape!=(self.vdim,) or not np.isfinite(k).all() or not np.isfinite(v).all():raise ValueError("invalid KV")
        old={i:self._decode(e) for i,e in self.entries.items()}
        self.count+=1;self.qk=self._basis(self.qk,k);self.qv=self._basis(self.qv,v);self.version+=1
        for i,e in self.entries.items():
            if not e["exact"]:
                e["k"],e["v"]=old[i][0]@self.qk,old[i][1]@self.qv
            e["basis_version"]=self.version
        self.entries.pop(key,None)
        exact_ids=[i for i,e in self.entries.items() if e["exact"]]
        exact=len(exact_ids)<self.anchors
        if not exact and self.anchors:
            worst=min(exact_ids,key=lambda i:self.entries[i]["priority"])
            if priority>self.entries[worst]["priority"]:
                e=self.entries[worst];e["k"],e["v"]=e["k"]@self.qk,e["v"]@self.qv;e["exact"]=False;exact=True
        self.entries[key]=dict(k=k.copy() if exact else k@self.qk,v=v.copy() if exact else v@self.qv,exact=exact,priority=float(priority),basis_version=self.version)
        while len(self.entries)>self.capacity:
            victim=next((i for i,e in self.entries.items() if not e["exact"]),next(iter(self.entries)))
            self.entries.pop(victim)
    def get(self,key):
        if key not in self.entries:return None
        self.entries.move_to_end(key)
        return self._decode(self.entries[key])
    def state_dict(self):
        return dict(kdim=self.kdim,vdim=self.vdim,capacity=self.capacity,rank=self.rank,anchors=self.anchors,eta=self.eta,count=self.count,version=self.version,qk=self.qk,qv=self.qv,entries=list(self.entries.items()))
    @classmethod
    def from_state(cls,d):
        obj=cls(d["kdim"],d["vdim"],d["capacity"],d["rank"],d["anchors"],d["eta"])
        for n in ("count","version","qk","qv"):setattr(obj,n,d[n])
        obj.entries=OrderedDict(d["entries"])
        for e in obj.entries.values():obj._decode(e)
        return obj
    def bytes(self):return self.qk.nbytes+self.qv.nbytes+sum(e["k"].nbytes+e["v"].nbytes for e in self.entries.values())
