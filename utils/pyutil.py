from einops.einops import rearrange
import numpy as np

def unzip(a):
    return tuple(map(list,zip(*a)))

# interable, comprabarable metric function -> max index, max element, max value
def argmax(li,func = lambda x: x):
    index, max_val,max_el = None,None,None
    for i,el in enumerate(li):
        val = func(el)
        if max_val is None or val > max_val:
            index, max_val,max_el = i, val,el
    return index,max_el,max_val

# interable, comprabarable metric function -> max index, max element, max value
def argmin(li,func = lambda x: x):
    ind,el,val = argmax(li,lambda x: -func(x))
    return ind,el,-val

def to_channel_first(images):
    return rearrange(images,'... r col c -> ... c r col')

def to_channel_last(images):
    return rearrange(images,'... c r col -> ... r col c')

def embed_mat(mat,size=4):
    if len(mat.shape) == 1:
        trans = np.ones((size,))
        trans[:mat.shape[0]] = mat
    else:
        trans = np.eye(size)
        trans[:mat.shape[0],:mat.shape[1]] = mat
    return trans

def dig(obj, *keys, default=None,raise_errors=False): 
    if len(keys) == 0:
        return obj
    if hasattr(obj,'__getitem__'):
        try:
            return dig(obj[keys[0]],*keys[1:],default=default)
        except (IndexError, KeyError) as e:
            if raise_errors:
                raise e
            else:
                return default
    else:
        return default

