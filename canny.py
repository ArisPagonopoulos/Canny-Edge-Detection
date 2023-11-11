import numpy as np 
from matplotlib import pyplot as plt 
import pandas as pd
from skimage import io
import matplotlib.cm as cm
from sys import argv


def zero_pad(image, pad_top, pad_down, pad_left, pad_right):
    """Creating padding around the image"""
    out = np.zeros((image.shape[0]+pad_top+pad_down, image.shape[1]+pad_left+pad_right))
    out[pad_top:image.shape[0]+pad_down, pad_left :image.shape[1] + pad_right] = image
    return out

def conv(image, kernel):
        """Convolution function
        image : numpy array of an image
        kernel: numpy array of the kernel"""
        Hi, Wi = image.shape
        Hk, Wk = kernel.shape
        out = np.zeros((Hi, Wi))
        kernel = np.flip(kernel)
        padded_img = zero_pad(image, int((Hk-1)/2), int((Hk-1)/2), int((Wk-1)/2), int((Wk-1)/2))
        for row in range(int((Hk-1)/2), Hi + int((Hk-1)/2)):
            for column in range(int((Wk-1)/2), Wi  + int((Wk-1)/2)):
                value = np.sum(padded_img[row - int((Hk-1)/2):row +int((Hk-1)/2) + 1, column - \
                int((Wk-1)/2): column+int((Wk-1)/2) + 1]*kernel)
                out[row - int((Hk-1)/2), column - int((Wk-1)/2)] = value
        return out

def gaussian2d(sig=None):
        """Gaussian filter application
        sig : scalar"""
        filter_size = int(sig * 6)
        if filter_size % 2 == 0:
            filter_size += 1     
        ax = np.arange(-filter_size // 2 + 1., filter_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
        return kernel / np.sum(kernel)

def smooth(image):
        """Application of gaussian smoothing to a given image
        image: numpy array of an image"""
        #δημιουργούμε τον πυρήνα καλώντας την συνάρτηση guassian2d για sig = 1.5
        kernel = gaussian2d(1)
        #επιστρέφουμε τον πίνακα συνέλιξης της εικόνας και του πυρήνα
        return conv(image, kernel)

image = io.imread("michael.jpg", as_gray=True)
smoothed = smooth(image)

def gradient(image):
        """Calculation of the magnitude and the degree of an image
        image: numpy array of an image"""
        kernel_y  = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
        #Gx, Gy kernel
        gx = conv(image, kernel_x)
        gy = conv(image, kernel_y)
        #magnitude
        g_mag = np.sqrt((gx**2)+(gy**2))
        g_theta = np.arctan2(gy, gx)
        return g_mag, g_theta

g_mag, g_theta = gradient(smoothed)

print("Magnitude and theta calculated")
def nms(g_mag, g_theta):
        """Non magnitude supression application
        g_mag : magnitude, scalar
        g_theta: scalar, radians"""
        #to_degrees
        g_theta_degrees = np.degrees(g_theta)
        #nearest multiple to 45
        ne = 45*np.round(g_theta_degrees/45)
        nms_response = g_mag.copy()
        #padding addition
        padded = zero_pad(nms_response, 1, 1, 1, 1)
        for row in range(1, g_mag.shape[0]+1):
            for column in range(1,g_mag.shape[1]+1):
                if (ne[row-1, column-1]) in [45,-45]:
                    if padded[row, column]<=max([padded[row-1, column+1],padded[row+1,column-1]]):
                        nms_response[row-1, column-1] = 0
                elif (ne[row-1, column-1]) in [90,-90]:
                    if padded[row, column]<=max([padded[row-1, column],padded[row+1,column]]):
                        nms_response[row-1, column-1] = 0
                elif (ne[row-1, column-1]) in [135,-135]:
                    if padded[row, column]<=max([g_mag[row-1, column-1],padded[row+1,column+1]]):
                        nms_response[row-1, column-1] = 0
    
                elif (ne[row-1, column-1]) in [0,180,-180]:
                    if padded[row, column]<=max([padded[row, column+1], padded[row, column-1]]):
                        nms_response[row-1, column-1] = 0
    
        return nms_response

nms_image = nms(g_mag, g_theta)

print("Non magnitude finished")


def bfs(mat, strong, t_min):
        """BFS algorithm for hysteresis thresholding"""
        visited = np.zeros_like(mat, dtype = bool)
        valid=[]
        while len(strong)>0:
            pixel = strong.pop()
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    if isValid(visited, mat, pixel[0]+i,pixel[1]+j,t_min):
                        strong.append((pixel[0]+i, pixel[1]+j))
                        valid.append((pixel[0]+i, pixel[1]+j))
            visited[pixel]=True
        return set(valid)

def isValid(vis, mat,row,col,t_min):
        if (row < 0 or col < 0 or row >=mat.shape[0] or col >=mat.shape[1]):
            return False
        if (vis[row, col]):
            return False
        if (mat[row, col]<t_min):
            return False
        return True

def hysteresis_threshold(image,g_theta, use_g_theta=False):
        """Hysteresis threshold application
        image : numpy array
        g_theta: direction in degrees, scalar"""
        t_max = 0.08*image.max()
        t_min = 0.20*t_max#0.25
        g_high = np.zeros_like(image)
        g_high_ind = np.transpose(np.nonzero(image > t_max))
        strong = []
        for i in g_high_ind:
            g_high[i[0], i[1]] = image[i[0], i[1]]
            strong.append((i[0], i[1]))
        val = bfs(image, strong, t_min)
        for i in val:
            g_high[i[0],i[1]]=image[i[0], i[1]]
        return g_high

#ploting the results
thresholded = hysteresis_threshold(nms_image,g_theta,False)
thresholded[thresholded>0]=255
print("Hysteresis thresholded finished")
plt.figure(figsize=(20,16))
plt.subplot(2,2,1)
plt.imshow(image, cmap = "gray")
plt.title("Grayscale")
plt.subplot(2,2,2)
plt.imshow(smoothed, cmap = "gray")
plt.title("Gaussian Smoothing")
plt.subplot(2,2,3)
plt.imshow(nms_image, cmap = "gray")
plt.title("Non-maximum supression")
plt.subplot(2,2,4)
plt.imshow(thresholded, cmap = "gray")
plt.title("Hysteresis")
plt.show()
print(thresholded)