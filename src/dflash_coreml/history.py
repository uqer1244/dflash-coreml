"""Guarded process-local deferred GDN history from the validated research path."""
import mlx.core as mx
from mlx_vlm.models.qwen3_5 import speculative_verifier as sv, gated_delta as gd
from mlx_vlm.models.cache import ArraysCache

_original_update=sv.gated_delta_update
_original_select=ArraysCache._select_speculative_states

class DeferredHistory:
    def __init__(self,q,k,v,g,beta,state,mask):
        self.factors=(q,k,v,g,beta);self.initial=state;self.mask=mask
        self.shape=(q.shape[0],q.shape[1]-1,*state.shape[1:])
        self.factor_bytes=sum(x.nbytes for x in self.factors)+(0 if mask is None else mask.nbytes)
    def replay(self,length):
        return kernel(*[x[:,:length] for x in self.factors],self.initial,None if self.mask is None else self.mask[:,:length])[1]

def kernel(q,k,v,g,beta,state,mask):
    B,T,Hk,Dk=k.shape;Hv,Dv=v.shape[2:]
    fn=gd._gated_delta_with_states_kernel if mask is None else gd._gated_delta_with_states_kernel_masked
    inputs=[q,k,v,g,beta,state,T]
    if mask is not None:inputs.append(mask)
    return fn(inputs=inputs,template=[('InT',q.dtype),('StT',state.dtype),('Dk',Dk),('Dv',Dv),('Hk',Hk),('Hv',Hv),('StateT',0)],grid=(32,Dv,B*Hv),threadgroup=(32,4,1),output_shapes=[(B,T,Hv,Dv),state.shape,(B,0,Hv,Dv,Dk)],output_dtypes=[q.dtype,state.dtype,state.dtype])

def update(q,k,v,a,b,A_log,dt_bias,state=None,mask=None,use_kernel=True,state_steps=None,cache=None,cache_index=1):
    txn=getattr(cache,'_speculation',None)
    if not (isinstance(cache,ArraysCache) and txn and q.shape[0]==1 and q.shape[1]>1 and use_kernel and state is None and state_steps is None and cache_index not in txn['records'] and txn['length']==q.shape[1] and mx.default_device()==mx.gpu):
        return _original_update(q,k,v,a,b,A_log,dt_bias,state,mask,use_kernel,state_steps,cache,cache_index)
    g,beta=gd._compute_g_beta(A_log,a,b,dt_bias)
    if g.ndim!=3:
        return _original_update(q,k,v,a,b,A_log,dt_bias,state,mask,use_kernel,state_steps,cache,cache_index)
    initial=cache[cache_index]
    if initial is not None and initial.shape[0]!=1:
        return _original_update(q,k,v,a,b,A_log,dt_bias,state,mask,use_kernel,state_steps,cache,cache_index)
    if initial is None:initial=mx.zeros((1,v.shape[2],v.shape[3],k.shape[3]),dtype=mx.float32)
    y,final,_=kernel(q,k,v,g,beta,initial,mask)
    history=DeferredHistory(q,k,v,g,beta,initial,mask)
    cache.record_speculative_states(cache_index,history,final);cache[cache_index]=final
    return y,final

def select(record,initial,lengths,total):
    history=record[1]
    if not isinstance(history,DeferredHistory):return _original_select(record,initial,lengths,total)
    assert len(lengths)==1
    keep=lengths[0]
    if keep==0:return initial
    if keep==total:return record[2]
    return history.replay(keep)
