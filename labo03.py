import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import skimage as sk


# Part 1
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