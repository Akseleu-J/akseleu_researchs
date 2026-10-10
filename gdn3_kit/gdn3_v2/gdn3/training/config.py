"""Serializable training schedule contract."""
from dataclasses import dataclass, asdict
import math

@dataclass(frozen=True)
class TrainConfig:
    steps:int=1000
    batch_size:int=16
    sequence_length:int=512
    learning_rate:float=3e-4
    min_learning_rate:float=3e-5
    warmup_steps:int=100
    weight_decay:float=0.1
    clip_norm:float=1.0
    ema_decay:float=0.999
    seed:int=0
    remat:bool=False
    mode:str="full_bptt"
    save_every:int=100
    def __post_init__(self):
        if min(self.steps,self.batch_size,self.sequence_length,self.save_every)<1:raise ValueError("steps/batch/sequence/save interval must be positive")
        if self.warmup_steps<0 or self.warmup_steps>=self.steps:raise ValueError("warmup must be in [0, steps)")
        if not self.learning_rate>0 or not 0<=self.min_learning_rate<=self.learning_rate:raise ValueError("invalid learning rates")
        if self.weight_decay<0 or self.clip_norm<=0 or not 0<=self.ema_decay<1:raise ValueError("invalid optimizer controls")
        if self.mode!="full_bptt":raise NotImplementedError("TBPTT is not implemented in this release")
    def to_dict(self):return asdict(self)
    def lr(self,step):
        if self.warmup_steps and step<self.warmup_steps:return self.learning_rate*(step+1)/self.warmup_steps
        progress=min(1.,max(0.,(step-self.warmup_steps)/max(1,self.steps-self.warmup_steps)))
        return self.min_learning_rate+.5*(self.learning_rate-self.min_learning_rate)*(1+math.cos(math.pi*progress))
