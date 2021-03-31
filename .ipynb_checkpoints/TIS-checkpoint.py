import matplotlib.pyplot as plt
import numpy as np

def showImage(Images, width=10, height=10, showGrid=True, HLines=None, VLines=None, w_label_step=0, h_label_step=0,  grid_step=1, titles = None, colorMap=None, Max=None, Min=None, saveto=None, colorBar=True, linesColor='red', linesWidth=1):
    """
    Displays an Image (grayscale or RGB)
    
    Parameters:
    -----------
    Images       : Image array (HxW or HxWx3) (multiple images in tuple OK)
    width        : Displayed image width (default 10)
    height       : Displayed images cluster height (when multiple images)
    showGrid     : display grid (default true)
    HLines       : array of vertical positions to highlight pixels
    VLines       : array of horizontal positions to highlight pixels
    w_label_step : width labels step
    h_label_step : height label step
    grid_step    : grid step
    titles       : figure titles (default none)
    colorMap     : colormap to apply (default gray when grey scale, ignored when RGB), default None
    colorBar     : Add color bar (default True)
    Max          : pixel max (default to 255 or 1 depending of data)
    Min          : pixel min (default 0)
    saveto       : path to save figure
    linesColor   : color of guidelines (default red)
    linesWidth   : width of lines (default 1)
    
    Returns:
    --------
    figure, ax (matplotlib)
    """
    maxPixelsPerWidthUnitMajor = 2.5
    maxPixelsPerWidthUnitMinor = 10
    
    if(type(Images) == tuple or type(Images) == list):
        if(len(Images) > 1 and len(Images) <= 2):
            imagesX = 2
            imagesY = 1
        elif(len(Images) > 2 and len(Images) <= 4):
            imagesX = 2
            imagesY = 2
        elif(len(Images) > 4 and len(Images) <= 9):
            imagesX = 3
            imagesY = 3
        
        imagesCount = len(Images)
    else:
        imagesX = 1
        imagesY = 1
        Images = [Images]
        imagesCount = 1
        if(not colorMap is None):
            colorMap = [colorMap]
        if(not titles is None):
            titles = [titles]
         
    if(imagesCount == 1):
        height = width/Images[0].shape[1]*Images[0].shape[0]
        
            
    fig, axs = plt.subplots(imagesY, imagesX, figsize=(width, height))

    i = 0
    for Image in Images:
        
        if(imagesCount == 1):
            ax = axs
        else:
            ax = axs.reshape(axs.size)[i]
        
        
        
        

        minImage = np.min(Image)
        maxImage = np.max(Image)

        if(Image.dtype == np.dtype('bool')):
            defaultMax = 1
            defaultMin = 0
        elif('int' in str(Image.dtype)):
            
            info = np.iinfo(Image.dtype)
            defaultMax = info.max
            defaultMin = info.min
        else:
            defaultMin = np.min(Image)
            defaultMax = np.max(Image)


        Max = defaultMax if Max is None else Max
        Min = defaultMin if Min is None else Min



        skips = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

        
        
        ax.set_xlabel("%d pixels" % Image.shape[1]);
        ax.set_ylabel("%d pixels" % Image.shape[0]);

        
        
        if(Image.ndim == 2):
            if(not colorMap is None):
                im = ax.imshow(Image, cmap= colorMap[i], vmin=Min, vmax=Max);
            else:
                im = ax.imshow(Image, cmap= 'gray', vmin=Min, vmax=Max);
            
            if(colorBar):
                fig.colorbar(im, ax=ax)
        else:
            im = ax.imshow(Image, vmin=Min, vmax=Max);
            
        if(not titles is None):
            ax.set_title(titles[i]);

        skipI = 0
        while(Image.shape[1] / width / skips[skipI] > maxPixelsPerWidthUnitMajor / np.max([imagesX, imagesY])):
            skipI += 1
        if(w_label_step > 0):
            ax.set_xticks(np.arange(0, Image.shape[1], w_label_step));
            ax.set_xticklabels(np.arange(0, Image.shape[1], w_label_step));
        else:
            ax.set_xticks(np.arange(0, Image.shape[1], skips[skipI]));
            ax.set_xticklabels(np.arange(0, Image.shape[1], skips[skipI]));

        if(h_label_step > 0):
            ax.set_yticks(np.arange(0, Image.shape[0], h_label_step));
            ax.set_yticklabels(np.arange(0, Image.shape[0], h_label_step));
        else:
            ax.set_yticks(np.arange(0, Image.shape[0], skips[skipI]));
            ax.set_yticklabels(np.arange(0, Image.shape[0], skips[skipI]));

        if(showGrid and (Image.shape[0] / height <= maxPixelsPerWidthUnitMinor / np.max([imagesX, imagesY]) or  grid_step > 1)):
            ax.set_xticks(np.arange(-0.5, Image.shape[1]+0.5,  grid_step), minor=True);
            ax.set_yticks(np.arange(-0.5, Image.shape[0]+0.5,  grid_step), minor=True);
            ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5);



        if(not HLines is None):
            if(np.isscalar(HLines)):
                ax.axhline(HLines, color=linesColor, linewidth=linesWidth);
            else:            
                for H in HLines:
                    ax.axhline(H, color=linesColor, linewidth=linesWidth);
        if(not VLines is None):
            if(np.isscalar(VLines)):
                ax.axvline(VLines, color=linesColor, linewidth=linesWidth);
            else:            
                for V in VLines:
                    ax.axvline(V, color=linesColor, linewidth=linesWidth);
        i += 1
        
    if(not saveto is None):
            fig.savefig(saveto);
    return fig, axs


#
# https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
#
def find_nearest(array, value, last=False):
    """
    Finds element in array closest to specified value
    
    Parameters:
    -----------
    array : data
    value : value to search for
    last  : find last element that satisfies condition (default: False)
    
    Returns:
    --------
    idx : element index
    val : value
    """
    array = np.asarray(array)
    if(last):
        idx = np.size(array)-1 - (np.abs(array[::-1] - value)).argmin() 
    else:
        idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def showHistogram(f, color='RoyalBlue', saveto=None):
    """
    Displays histograms and its cumulative for an image f
    Parameters:
    -----------
        f : image
        color : line color (default RoyalBlue)
        saveto : save path (default None)
    
    Returns:
    --------
        fig : figure
        axs : axis   
    """
    
    
    if('int' in str(f.dtype)):
        info = np.iinfo(f.dtype)
        inMin = info.min
        inMax = info.max
    else:
        inMin = np.min(f)
        inMax = np.max(f)
    
    h = computeHisto(f)
    hX = np.linspace(inMin, inMax, h.size)
    
    stemLimit = 20
    hc, hcn = computeCumulativeHisto(h)
    
    fig, axs = plt.subplots(1,3,figsize=(20,5))
    if(len(h) > stemLimit):
        axs[0].plot(hX, h, c=color)
    else:
        axs[0].stem(hX, h, use_line_collection=True, c=color)
    axs[0].grid()
    axs[1].plot(hX, hc, c=color)
    axs[2].plot(hX, hcn, c=color)
    axs[1].grid()
    axs[2].grid()
    axs[1].set_xlim(inMin,inMax)
    axs[1].set_ylim(0,np.max(hc)+1)
    axs[2].set_xlim(inMin,inMax)
    axs[2].set_ylim(0,np.max(hcn)+1)
    axs[0].set_title('Histogramme')
    axs[1].set_title('Histogramme cumulé')
    axs[2].set_title('Histogramme cumulé normalisé')
    [ax.grid() for ax in axs];
    
    if(not saveto is None):
            fig.savefig(saveto);
            
    return fig, axs

def computeHisto(f):
    """
    Compute histogram of image
    Parameters:
    -----------
        f : image
    Returns:
    --------
        h : histogram
    """
    
    if('int' in str(f.dtype)):
        info = np.iinfo(f.dtype)
        inMax = info.max
        inMin = info.min
    else:
        inMin = int(np.floor(np.min(f)))
        inMax = int(np.ceil(np.max(f)))
        
    if('float' in str(f.dtype)):
        N = 256
        rng = np.linspace(inMin, inMax, N)
    else:
        N = inMax - inMin + 1
        rng = np.arange(inMin, inMax+1)
    h = np.zeros(N, dtype=int)
    for i in range(0, N):
        h[i] = np.sum(f == rng[i])
    return h

def imgLevelAdjust(f, _mini=1, _maxi=99, outDType='uint8'):
    """
    Adjusts image's contrast  TODO: make it generic for image type
    Parameters:
    -----------
        f     : image
        _mini : lower percentage limit (default 1)
        _maxi : higher percentage limit (default 99)
        
    Returns:
    --------
        H : adjusted image
    """
    outMin = np.iinfo(outDType).min
    outMax = np.iinfo(outDType).max
    inMin  = np.iinfo(f.dtype).min
    inMax  = np.iinfo(f.dtype).max
    
    
    h = computeHisto(f)
    _, hc = computeCumulativeHisto(h)
    hc[0] = 0
    minIndex, _ = find_nearest(hc, _mini*255/100, last=True)
    maxIndex, _ = find_nearest(hc, _maxi*255/100, last=False)
    lut = np.zeros(inMax-inMin, dtype='uint8')
    lut[minIndex:maxIndex] = outMax/(maxIndex-minIndex) * np.arange(maxIndex-minIndex)
    lut[maxIndex:] = outMax
    
    return applyLUT(f, lut).astype(outDType)
    

def applyLUT(f,lut):
    """
    Applies LUT to an image
    
    Parameters:
    -----------
        f   : image
        lut : lookup table
    Returns:
    -----------
        new image
    """
    info = np.iinfo(f.dtype)
    g = lut[f+info.min]
    return g

def computeCumulativeHisto(h):
    """
    Computes cumulative histogram from base histrogram
    
    Parameters:
    -----------
        h : histogram
        
    Returns:
    --------
        cumulative histogram,
        cumulative normalized histogram
    """
    hc = np.zeros(len(h), dtype=h.dtype)
    hc[0] = h[0]
    for i in range(1, len(h)):
        hc[i] += hc[i-1] + h[i]
    hn = (hc/np.max(hc)*255).astype(hc.dtype)
    return hc, hn

def halfToning(img):
    """
    Applies halftoning algorithm to f and returns the results as 2D array
    
    Parameters:
    ----------
        img : image
    Returns:
    -------
        h : new image
    """
    if(img.ndim > 2):
        image = img[:,:,0]
    else:
        image = img
    
    threshold = filters.threshold_otsu(image)
    
    height = image.shape[0]
    width = image.shape[1]
    
    
    f = image.copy().astype('int16')
    
    h = np.zeros((height, width), dtype='bool')
    
    mask = 1/16*np.array([
        [0, 0, 7],
        [3, 5, 1]
    ])
    
    maskHeight = mask.shape[0]
    maskWidth = mask.shape[1]
    
    lines = np.arange(height)
    columns = np.arange(width)
    
    for l in lines:
        for c in columns:
            h[l,c] = f[l,c] >= threshold
            e = f[l,c] - h[l,c]*255
            #print(f)
            
            for l1 in np.arange(maskHeight):
                for c1 in np.arange(maskWidth):
                    if(l1 > 0 or c1 > 1):
                        if(l+l1 < height and c+c1-1 < width):
                            #print(e)
                            #print("+",(e * mask[l1,c1]))
                            f[l+l1,c+c1-1] += (e * mask[l1,c1]).astype('int16')
                            pass
    return h


#
# PADDINGS
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
    print(f)
    g = np.zeros((h, w))
    
    g = f[b:h+b,a:w+a]
    print(g)
    return g

#
# Labo04 - Part 2
#
def conv2D(_f,_mask, norm=True):
    """
    2D convolution of an image
    
    Parameters:
    -----------
        _f    : input image
        _mask : mask
        norm  : normalize image 0-255 (default true)
    Returns:
    --------
        g     : new image
    """
    #f('int' in str(_f.dtype)):
    #    info = np.iinfo(_f.dtype)
    #    outMin = info.min;
    #    outMax = info.max;
    #else:
    #    outMin = 0;
    #    outMax = 255;
        
    a = _mask.shape[1]//2
    b = _mask.shape[0]//2
    
    height = _f.shape[0]
    width = _f.shape[1]
    lines = np.arange(height)
    columns = np.arange(width)
    
    f = imagePadding(_f, _mask, 'zero')
    g = np.zeros_like(_f, 'int16')
    
    for l in lines:
        for c in columns:
            hm = f[l:l+2*b+1, c:c+2*a+1].astype('int16')
            v = np.sum(hm * _mask).astype('int16')
            #print(v)
            g[l,c] = v
    if(norm):
        g = imgLevelAdjust(g, 0, 100, outDType='uint8')
    
    return g