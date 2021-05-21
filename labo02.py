import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from labo01 import mapping

#
# Labo04 - Part 1
#
def circularPadding(f,m):
    """
    Adds circular-type padding to an image so that mask can be applied
    Parameters:
    -----------
        f : image
        m : mask
        
    Returns:
    --------
        fp : padded image
    """
    # m : 2b+1 x 2a+1
    a = m.shape[1]//2
    b = m.shape[0]//2 
    h = f.shape[0]
    w = f.shape[1]
    
    fp = np.zeros((h+2*b, w+2*a))
    # f -> [f f f]
    K = np.concatenate([f,f,f])
    #             [[ f f f ]
    # [f f f] ->  [ f f f ]
    #             [ f f f ]]
    K = np.concatenate([K,K,K], axis=1)
    
    #3h x 3w -> h+2b x 2+2a
    fp = K[h-b:2*h+b,w-a:2*w+a]
    
    return fp;
def mirrorPadding(f,m):
    """
    Adds mirror-type padding to an image so that mask can be applied
    Parameters:
    -----------
        f : image
        m : mask
        
    Returns:
    --------
        fp : padded image
    """
    # m : 2b+1 x 2a+1
    a = m.shape[1]//2
    b = m.shape[0]//2 
    h = f.shape[0]
    w = f.shape[1]
    
    fp = np.zeros((h+2*b, w+2*a))
    f2 = np.flip(f, 1)
    f3 = np.flip(f, 0)
    f1 = np.flip(f3, 1)
    
    #      [f1 f3 f1]
    #K -> [f2 f  f2]
    #      [f1 f3 f2]
    
    #A = [f1 f3 f1]
    A = np.concatenate([f1, f3, f1], axis=1)
    #B = [f2 f f1]
    B = np.concatenate([f2, f, f2], axis=1)
    C = np.concatenate([f1, f3, f1], axis=1)
    #   [[A]
    #K = [B]
    #    [C]]
    K = np.concatenate([A,B,C], axis=0)
    
    fp = K[h-b:2*h+b,w-a:2*w+a]
    
    
    return fp;
def replicatePadding(f,m):
    """
    Adds replicate-type padding to an image so that mask can be applied
    Parameters:
    -----------
        f : image
        m : mask
        
    Returns:
    --------
        fp : padded image
    """
    # m : 2b+1 x 2a+1
    a = m.shape[1]//2
    b = m.shape[0]//2 
    h = f.shape[0]
    w = f.shape[1]
    
    fp = np.zeros((h+2*b, w+2*a),dtype=f.dtype)
    #     [C11 lt C12]
    #fp = [cl  f  cr]
    #     [C21 lb C22]
    C11 = np.ones((b,a),dtype=f.dtype) * f[0,0]
    C12 = np.ones((b,a),dtype=f.dtype) * f[0,-1]
    C21 = np.ones((b,a),dtype=f.dtype) * f[-1,0]
    C22 = np.ones((b,a),dtype=f.dtype) * f[-1,-1]
    
    lt = np.ones((b,w),dtype=f.dtype)  * f[0,:]
    lb = np.ones((b,w),dtype=f.dtype)  * f[-1,:]
    cl = np.ones((h, a),dtype=f.dtype) * np.transpose([f[:,0]])
    cr = np.ones((h, a),dtype=f.dtype) * np.transpose([f[:,-1]])
    
    A = np.concatenate([C11, lt, C12], axis=1)
    B = np.concatenate([cl, f, cr], axis=1)
    C = np.concatenate([C21, lb, C22], axis=1)
    fp = np.concatenate([A,B,C], axis=0)
    
    return fp;
def zeroPadding(f,m):
    """
    Adds zero padding to an image so that mask can be applied
    Parameters:
    -----------
        f : image
        m : mask
        
    Returns:
    --------
        fp : padded image
    """
    # m : 2b+1 x 2a+1
    a = m.shape[1]//2
    b = m.shape[0]//2 
    h = f.shape[0]
    w = f.shape[1]
    
    fp = np.zeros((h+2*b, w+2*a),dtype=f.dtype)
    #     [C  UD C]
    #fp = [LR f  LR]
    #     [C  lb C]
    C = np.zeros((b,a),dtype=f.dtype)
    
    UD = np.zeros((b,w),dtype=f.dtype)
    LR = np.zeros((h,a),dtype=f.dtype)
    A = np.concatenate([C, UD, C], axis=1)
    B = np.concatenate([LR, f, LR], axis=1)
    fp = np.concatenate([A,B,A], axis=0)
    
    return fp;
def imagePadding(_f, _mask, _type='mirror'):
    """
    Adds specified padding to an image so that mask can be applied
    Parameters:
    -----------
        f : image
        m : mask
        _type : 'mirror', 'zero', 'replicate', 'circular' (default mirror)
        
    Returns:
    --------
        fp : padded image
    """
    # m : 2b+1 x 2a+1
    if(_type == 'mirror'):
        fp = mirrorPadding(_f, _mask)
    elif(_type == 'zero'):
        fp = zeroPadding(_f, _mask)
    elif(_type == 'replicate'):
        fp = replicatePadding(_f, _mask)
    elif(_type == 'circular'):
        fp = circularPadding(_f, _mask)
    
    return fp
def imageUnpadding(f,mask):
    # m : 2b+1 x 2a+1
    a = mask.shape[1]//2
    b = mask.shape[0]//2 
    h = f.shape[0] - 2*b
    w = f.shape[1] - 2*a
    g = np.zeros((h, w))
    
    g = f[b:h+b,a:w+a]
    return g

#
# Labo04 - Part 2
#
def conv2D(_f,_mask, norm=True, padding='zero'):
    """
    2D convolution of an image
    
    Parameters:
    -----------
        _f    : input image
        _mask : mask
        norm  : normalize image 0-255 (default true)
        padding : type of padding ('zero', 'mirror', 'replicate', 'circular') default 'zero'
    Returns:
    --------
        g     : new image
    """        
    a = _mask.shape[1]//2
    b = _mask.shape[0]//2
    
    height = _f.shape[0]
    width = _f.shape[1]
    lines = np.arange(height)
    columns = np.arange(width)
    
    f = imagePadding(_f, _mask, padding)
    g = np.zeros_like(_f, 'float')
    
    for l in lines:
        for c in columns:
            hm = f[l:l+2*b+1, c:c+2*a+1].astype('float')
            v = np.sum(hm * _mask)
            g[l,c] = v
    if(norm):
        g = mapping(g, np.min(g), np.max(g), 0, 255)
    
    return g
    
    
#
# Labo04 - Part 3
#
def gkern(kernlen=7, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel