import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
from labo01 import find_nearest
from labo02 import zeroPadding


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