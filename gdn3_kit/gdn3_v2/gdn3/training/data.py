"""Deterministic causal synthetic next-token stream and MQAR facts."""
import jax
import jax.numpy as jnp

def random_lm_batch(key,batch_size,sequence_length,vocab):
    if min(batch_size,sequence_length,vocab)<2:raise ValueError("batch, sequence, vocab must be >=2")
    return jax.random.randint(key,(batch_size,sequence_length),0,vocab,dtype=jnp.int32)

def mqar_batch(key,batch_size,sequence_length,vocab,n_facts=8):
    """Diagnostic generated recall data: key/value pairs are placed before a query.

    This is a scaffold, not a paper benchmark; splits and distractors must be
    specified by an experiment manifest before any quality claim.
    """
    if vocab<8 or sequence_length<8 or n_facts<1:raise ValueError("MQAR dimensions too small")
    tokens=jax.random.randint(key,(batch_size,sequence_length),4,vocab,dtype=jnp.int32)
    keys=jnp.arange(1,n_facts+1,dtype=jnp.int32)
    vals=(keys+n_facts)%(vocab-4)+4
    slots=jnp.linspace(1,sequence_length-4,n_facts,dtype=jnp.int32)
    for i in range(n_facts):tokens=tokens.at[:,slots[i]].set(keys[i]);tokens=tokens.at[:,slots[i]+1].set(vals[i])
    qpos=jnp.asarray(sequence_length-2,jnp.int32);which=jax.random.randint(key,(batch_size,),0,n_facts)
    tokens=tokens.at[:,qpos].set(keys[which]);tokens=tokens.at[:,-1].set(vals[which])
    return tokens,jnp.ones_like(tokens,dtype=bool),dict(query_position=qpos,keys=keys,values=vals,split="diagnostic_only")
