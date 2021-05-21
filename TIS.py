import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
from skimage.filters import threshold_otsu
from skimage import morphology
import scipy.stats as st




def showImage(Images, width=10, height=10, showGrid=True, HLines=None, VLines=None, w_label_step=0, h_label_step=0,  grid_step=1, titles = None, colorMap=None, Max=None, Min=None, saveto=None, colorBar=False, linesColor='red', linesWidth=1):
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

        if(Image.dtype == np.dtype('bool')):
            defaultMax = 1
            defaultMin = 0
        elif(Image.dtype == np.dtype('uint8')):
            defaultMax = 255
            defaultMin = 0
        else:
            defaultMin = np.min(Image)
            defaultMax = np.max(Image)


        axMax = defaultMax if Max is None else Max
        axMin = defaultMin if Min is None else Min



        skips = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

        
        
        ax.set_xlabel("%d pixels" % Image.shape[1])
        ax.set_ylabel("%d pixels" % Image.shape[0])

        
        
        if(Image.ndim == 2):
            if(not colorMap is None):
                im = ax.imshow(Image, cmap= colorMap[i], vmin=axMin, vmax=axMax)
            else:
                im = ax.imshow(Image, cmap= 'gray', vmin=axMin, vmax=axMax)

            if(colorBar):
                fig.colorbar(im, ax=ax)
        else:
            im = ax.imshow(Image, vmin=axMin, vmax=axMax)
            
        if(not titles is None):
            ax.set_title(titles[i])

        skipI = 0
        while(Image.shape[1] / width / skips[skipI] > maxPixelsPerWidthUnitMajor / np.max([imagesX, imagesY])):
            skipI += 1
        if(w_label_step > 0):
            ax.set_xticks(np.arange(0, Image.shape[1], w_label_step))
            ax.set_xticklabels(np.arange(0, Image.shape[1], w_label_step))
        else:
            ax.set_xticks(np.arange(0, Image.shape[1], skips[skipI]))
            ax.set_xticklabels(np.arange(0, Image.shape[1], skips[skipI]))

        if(h_label_step > 0):
            ax.set_yticks(np.arange(0, Image.shape[0], h_label_step))
            ax.set_yticklabels(np.arange(0, Image.shape[0], h_label_step))
        else:
            ax.set_yticks(np.arange(0, Image.shape[0], skips[skipI]))
            ax.set_yticklabels(np.arange(0, Image.shape[0], skips[skipI]))

        if(showGrid and (Image.shape[0] / height <= maxPixelsPerWidthUnitMinor / np.max([imagesX, imagesY]) or  grid_step > 1)):
            ax.set_xticks(np.arange(-0.5, Image.shape[1]+0.5,  grid_step), minor=True)
            ax.set_yticks(np.arange(-0.5, Image.shape[0]+0.5,  grid_step), minor=True)
            ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)



        if(not HLines is None):
            if(np.isscalar(HLines)):
                ax.axhline(HLines, color=linesColor, linewidth=linesWidth)
            else:            
                for H in HLines:
                    ax.axhline(H, color=linesColor, linewidth=linesWidth)
        if(not VLines is None):
            if(np.isscalar(VLines)):
                ax.axvline(VLines, color=linesColor, linewidth=linesWidth)
            else:            
                for V in VLines:
                    ax.axvline(V, color=linesColor, linewidth=linesWidth)
        i += 1
        
    if(not saveto is None):
            fig.savefig(saveto)
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
    [ax.grid() for ax in axs]
    
    if(not saveto is None):
            fig.savefig(saveto)
            
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
    
    assert str(f.dtype) == 'uint8', "Cannot compute histogram of non-uint8 image"
    info = np.iinfo(f.dtype)
    inMax = info.max
    inMin = info.min

    N = inMax - inMin + 1
    rng = np.arange(inMin, inMax+1)
    h = np.zeros(N, dtype=int)
    for i in range(0, N):
        h[i] = np.sum(f == rng[i])
    return h


def mapping(X, inMin, inMax, outMin, outMax, restrict=True):
    """
    Maps X's range to Y's range
        
    Parameters:
    -----------
        X : input array
        inMin : input array min
        inMax : input array max
        outMin : output array min
        outMax : output array max
        restict : force output range (default True)
    Returns:
    --------
        Y : output array
    """
    inMin = float(inMin)
    inMax = float(inMax)
    inMax = float(inMax)
    inMax = float(inMax)
    Y = (X - inMin) * (outMax - outMin) / (inMax - inMin) + outMin
    Y[Y < outMin] = outMin
    Y[Y > outMax] = outMax
    return Y.astype(X.dtype)

def imgLevelAdjust(f, _mini=1, _maxi=99, outDType=None, inMin=0, inMax=255, outMax=0, outMin=0):
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
    assert "int" in str(f.dtype), "Cannot adjust non-int matrices"
    info = np.iinfo(f.dtype)
    
    if(outMax != 0 or outMin != 0):
        #if specified, we use these
        pass
    elif(not outDType is None and "int" in str(outDType)):
        #If not specified and requested type is int, we find it's range
        outMin = np.iinfo(outDType).min
        outMax = np.iinfo(outDType).max
        
    
    h = computeHisto(f)
    _, hc = computeCumulativeHisto(h)
    hc[0] = 0
    minIndex, minValue = find_nearest(hc, _mini*inMin/100, last=True)
    maxIndex, maxValue = find_nearest(hc, _maxi*inMax/100, last=False)
    
    return mapping(f, minValue, maxValue, info.min, info.max).astype(f.dtype);

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
    assert np.min(f) >= 0, "Cannot apply LUT to negative values"
    g = lut[f]
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
    
    threshold = threshold_otsu(image)
    
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

#
# Labo05 - Part 1
#
def sedisk(radius=1):
    """
    Returns a matrix of 2*radius + 1 x 2*radius + 1 containing a disk with radius=radius
    Parameters:
    -----------
        radius
        
    Returns:
    --------
        out : disk matrix
    
    
    """
    l = 1+2*radius
    
    V = np.array([np.abs(np.arange(-radius,radius+1)), ]*l)
    
    #Distances
    D = V**2 + V.transpose()**2    
    
    out = np.zeros((l, l), dtype=bool)
    
    out[D+1 <= radius**2+radius] = True

    return out

def secross(radius=1):
    """
    Returns a matrix of 2*radius + 1 x 2*radius + 1 containing a cross with it's branch's size=radius
    Parameters:
    -----------
        radius
        
    Returns:
    --------
        out : cross matrix
    
    
    """
    l = 1+2*radius
    out = np.zeros((l, l), dtype=bool)
    for col in np.arange(-radius, radius+1):
        for lin in np.arange(-radius, radius+1):
            if(np.abs(col) + np.abs(lin) <= radius):
                out[lin + radius, col+radius] = 1     
    return out

def sebox(radius=1):
    """
    Returns a 1-filled square matrix of size 2*radius + 1
    """
    return np.ones((2*radius+1, 2*radius+1))

def erosion(f,se):
    """
    Erode image f with se
    
    Parameters:
    -----------
        f : input
        se : structured element
        
    Returns:
    --------
        out : output
    """
    
    out = sk.morphology.erosion(f, se)
    
    return out

def dilation(f,se):
    """
    Dilate image f with se
    
    Parameters:
    -----------
        f : input
        se : structured element
        
    Returns:
    --------
        out : output
    """
    
    return morphology.dilation(f, se)

def closing(f,se):
    """
    Closing operation (filling the gaps)
    
    Parameters:
    -----------
        f : input
        se : structured element
        
    Returns:
    --------
        output    
    """
    return erosion(dilation(f, se), se)

def opening(f,se):
    """
    Opening operation (smooth + narrows)
    
    Parameters:
    -----------
        f : input
        se : structured element
        
    Returns:
    --------
        output    
    """
    return dilation(erosion(f, se), se)

def white_tophat(f,se):
    """
    Top-hat transformation
    
    Parameters:
    -----------
        f : input
        se : structured element
        
    Returns:
    --------
        output
    """
    return f - opening(f, se)

def black_tophat(f,se):
    """
    Bottom-hat transformation
    
    Parameters:
    -----------
        f : input
        se : structured element
        
    Returns:
    --------
        output
    """
    return closing(f, se) - f


#
# Labo04 - Part 1
#

def drawLine(image, x0, y0, x1, y1, color=255, accumulate=False, inplace=False):
    """
    draws a line on a image from (x0, y0) -- (x1, y1)
    Parameters:
    -----------
        image : Image
        x0 : starting point
        y0 : 
        x1 : ending point
        y1 :
        color : value to print (default 255)
        accumulate : adds to the image (default False)
        inplace : edits the image directly (default False)
        
    Returns:
    --------
        f : new image
    """
    y, x = sk.draw.line(y0, x0, y1, x1)
    if(inplace):
        f = image
    else:
        f = image.copy()

    if(accumulate):
        f[y,x] += 1
    else:
        f[y,x] = color 

    return f


def drawCircle(image, radius, x_center, y_center, color=255, accumulate=False, inplace=False, fill=False):
    """
    draws a circle on a image with center at (x_center, y_center) and with radius radius
    Parameters:
    -----------
        image
        radius
        x_center
        y_center
        color : color to print (default 255)
        accumulate : adds to image instead of replacing
        inplace : works on the image directly  (default false)
        fill : fill (disc)
    
    Returns:
    --------
        f : new image
    """
    if(inplace):
        f = image
    else:
        f = image.copy()

    if(fill):
        y, x = sk.draw.disk((y_center, x_center), radius)
    else:
        y, x = sk.draw.circle_perimeter(y_center, x_center, radius)

    if(accumulate):
        f[y, x] += 1
    else:
        f[y, x] = color
    return f

def drawRectangle(image, x0, y0, x1, y1, fill=False, color=255):
    """
    draws a connector (line with terminations) on a image from (x0, y0) -- (x1, y1)
    
    Parameters:
    -----------
        image : Image
        x0 : starting point
        y0 : 
        x1 : ending point
        y1 : 
        start : starting node [None, 'circle', 'square', 'filled-square']
        end : endping node [None, 'circle', 'square', 'filled-square']
        color : color to print (default 255)
    Returns:
    --------
        f : new image
    """
    f = image.copy()
    if(fill):
        f[x0:x1+1, y0:y1+1] = color
    else:
        points = [[x0, y0], [x1,y0], [x1,y1], [x0,y1], [x0,y0]]
        for p in range(0, len(points)-1):
            f = drawLine(f, points[p][0], points[p][1], points[p+1][0], points[p+1][1], color)
    return f

def rhoTheta(x, y):
    """
    Returns p and theta from a pair of (xi,yi) points
    Parameters:
    -----------
        x
        y
    Returns:
    --------
        rho
        theta
    """
    if(x == 0 and y == 0):
        rho = 0
        theta = 0
    else:
        rho = np.sqrt(x**2 + y**2)
        theta = 2*np.arctan(y/(x+rho))
    return rho, theta

def computeIntersectFromPolar(rho1,theta,H,W):
    """
    Computes intersection with image borders from rho and theta
    Parameters:
    -----------
        rho1 : rho 
        theta : theta (degrees)
        H : image height
        W : image width
    Returns:
    --------
        P1 : point 1 (tuple)
        P2 : point 2 (tuple)
    """

    if(theta == 0):
        VL = -1
        HD = rho1
        HU = rho1
        VR = -1
    else:
        thetaSign = 1 if theta >= 0 else -1
        
        #intersection Vertical Left (theta < 0 -> VL < 0)
        if(np.equal(theta, 90)):
            VL = rho1
        else:
            VL = int(np.round(rho1/np.cos(np.pi/2-np.deg2rad(theta))))

        #intersection Horizontal Down
        if(np.equal(theta, 90)):
            HD = -1
        else:
            HD = int(np.round(-(H-1-VL)/np.tan(np.pi/2-np.deg2rad(theta))))
        
        #intersection Horizontal Up
        HU = int(np.round(rho1/np.cos(np.deg2rad(theta))))

        #intersection Vertical Right
        VR = int(np.round(-(W-1-HU)/np.tan(np.deg2rad(theta))))
        
    P1 = None
    P2 = None
    # 4 possible points, we search all of them to find 2 differents and valid
    for P in [(W-1, VR), (HD, H-1), (0, VL), (HU, 0)]:
        # if the point is valid
        if(P[0] >= 0 and P[0] < W and P[1] >= 0 and P[1] < H):
            if(P1 is None):
                P1 = P
            elif(P != P1): #the second must be different
                P2 = P
    return P1, P2  

    #
    # Labo04 - Part 2
    # 
class hough:
    """
    Hough space class
    """
    def __init__(self, f, thetaStep=1):
        """
        sets the base image for the hough space as well as the step for theta (1 degree by default)
        """
        self.f = f.copy()
        self.thetaStep = thetaStep
        self._calc()
    
    def _calc(self):
        """
        calculates hough space representation from image f
        """
        self.D = int(np.sqrt(self.f.shape[0]**2 + self.f.shape[1]**2))
        #heights : -D -> D
        self.heights = np.arange(-self.D, self.D+1)
        #thetas : -90 -> 90-step
        self.thetas = np.arange(-90, 90, self.thetaStep)
    
        self.space = np.zeros((self.heights.size, self.thetas.size), dtype='uint64')

        y, x = np.where(self.f)
        
        for i in np.arange(x.size):
            xc = x[i]
            yc = y[i]

            rhos = np.round(xc*np.cos(np.deg2rad(self.thetas)) + yc*np.sin(np.deg2rad(self.thetas)) - np.min(self.heights)).astype('int')
            thetasIndices = np.round(self.thetas - np.min(self.thetas)).astype('int')
            self.space[rhos, thetasIndices] += 1

    def addPoint(self, x, y):
        """
        Adds a point to the current image and re-calculates the hough space
        """
        assert x < self.f.shape[1], "x coordinate too big"
        assert y < self.f.shape[0], "y coordinate too big"

        self.f[y, x] = 255

        self._calc()


    def _indexToRho(self, index):
        """
        matrix index to rho value
        """
        rho = self.heights[index]
        return rho

    def _rhoToIndex(self, rho):
        """
        rho value to matrix index
        """
        index, _ = find_nearest(self.heights, rho)
        return index

    def _indexToTheta(self, index):
        """
        matrix index to theta value
        """
        theta = self.thetas[index]
        return theta
    
    def _thetaToIndex(self, theta):
        """
        theta value to matrix index
        """
        index, _ = find_nearest(self.thetas, theta)
        return index

    def findMax(self):
        """
        finds the max of the current hough space
        """
        y, x = np.where(self.space == np.max(self.space))
        return np.mean(self._indexToRho(y)), np.mean(self._indexToTheta(x))

    def findMaxs(self, count, splashSize = 5):
        """
        Recursively finds the maxs of the current hough space (hiding the current one under a black splash)
        Parameters:
        -----------
            count : number of maxs to find
            splashSize : size of "splash" to ignore the last maximum found (default 5)
        Returns:
        --------
            points : tuples of maximas locations
        """
        space_temp = self.space.copy()
        points = []
        for i in range(0, count):
            rho, theta = self.findMax()
            points.append((rho, theta))
            drawCircle(self.space, splashSize, self._thetaToIndex(theta), self._rhoToIndex(rho), inplace=True, color=0, fill=True)
        self.space = space_temp
        return points


        
    def image(self):
        """
        Returns the current image
        """
        return self.f

    def space(self):
        """
        Returns the current space
        """
        return self.space


    def plot(self, width=8, height=10, saveto=None):
        """
        Plots the hough space, can be saved at saveto
        """
        fig, ax = plt.subplots(figsize=(width,height))
        ax.imshow(self.space, cmap='Greys_r', interpolation='none', aspect='auto')
        
        xTickStep = 45
        yTIckStep = np.round(self.space.shape[0] / height / 10) * 10

        #x axis
        my_xticks = np.arange(-90, 90, xTickStep, dtype='int').astype('str')
        x = np.arange(0, self.thetas.size, xTickStep).astype('int')
        ax.set_xticks(x)
        ax.set_xticklabels(my_xticks)
        #y axis
        my_yticks = np.arange(-round(self.D/10)*10, self.D, yTIckStep, dtype='int').astype('str')
        y = np.arange(0, self.heights.size, yTIckStep).astype('int')
        ax.set_yticks(y)
        ax.set_yticklabels(my_yticks)
        ax.invert_yaxis()


        ax.grid()
        ax.set_title("Hough space accumulator")
        ax.set_ylabel("$\\rho$", fontsize=18)
        ax.set_xlabel("$\\theta$", fontsize=18)
        if(not saveto is None):
            fig.savefig(saveto)

    def plot3D(self, width=12, height=12, saveto=None):
        """
        plots the current hough space in 3D, can be saved at saveto
        """
        fig = plt.figure(figsize=(width,height))
        ax = fig.add_subplot(projection='3d')
        X, Y = np.meshgrid(self.thetas, self.heights)
        Z = self.space
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        if(not saveto is None):
            fig.savefig(saveto)

def drawLineRhoTheta(f, rho, theta, inplace=False, color=255):
    """
    Draws a line on a image with rho and theta
    Parameters:
    -----------
        f     : Image
        rho   : rho
        theta : theta
        inplace : draws on the image directly (default False)
        color : color of line
    Returns:
    --------
        f : new image
    """

    P1, P2 = computeIntersectFromPolar(rho, theta, f.shape[0], f.shape[1])

    return drawLine(f, P1[0], P1[1], P2[0], P2[1], inplace=inplace, color=color)

#
# Labo04 - Part3
#
def accumulateCircles(image, radius):
    """
    Accumulates a circle or radius "radius" at each lit pixel in image
    Parameters:
    -----------
        image : image
        radius : radius
    Returns:
    --------
        f : new image
    """
    f = zeroPadding(image, np.zeros((2*radius+1, 2*radius+1)))

    y, x = np.where(f)

    for i in np.arange(x.size):
        drawCircle(f, radius, x[i], y[i], accumulate=True, inplace=True)
    return f