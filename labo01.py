import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu




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