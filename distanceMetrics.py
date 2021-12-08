import numpy as np
from numpy.core.umath_tests import inner1d
from utilities import showImage,translateArray


blurKernelSize = 7


def blurredDistance(a,b, show = False):
    import cv2
    kernelSize = blurKernelSize
    
    a = cv2.GaussianBlur(a,(kernelSize,kernelSize),sigmaX = 0)
    b = cv2.GaussianBlur(b,(kernelSize,kernelSize),sigmaX = 0)
    
    if show:
        showImage(a)
        showImage(b)
    return -np.sum(np.abs(a - b))



def asymmetricBlurredDistance(a,b, show = False, kernelSize = None, factor = 2, invariance = 0):
    # a = target
    # b = current
    # if you see a pixel in current that isn't in target, that's really bad
    # if you see a pixel and target that isn't an current, that's not so bad
    import cv2
    kernelSize = blurKernelSize if kernelSize == None else kernelSize

    # threshold the images
    a = np.copy(a*2)
    a[a > 0.35] = 1.0
    a[a <= 0.35] = 0.0
#    showImage(a)

    b = np.copy(b)
    b[b > 0.5] = 1.0
    b[b <= 0.5] = 0.0
#    showImage(b)
    
    a = cv2.GaussianBlur(a,(kernelSize,kernelSize),sigmaX = 0)
    b = cv2.GaussianBlur(b,(kernelSize,kernelSize),sigmaX = 0)
    
    if show:
        showImage(a)
        showImage(b)

    bestDistance = None
    PERTURBATIONINVARIANCE = invariance
    for dx in range(2*PERTURBATIONINVARIANCE + 1):
        for dy in range(2*PERTURBATIONINVARIANCE + 1):
            d = translateArray(a,dy - PERTURBATIONINVARIANCE, dx - PERTURBATIONINVARIANCE) - b
            positives = d > 0
            targetBigger = np.sum(np.abs(d[d > 0]))
            currentBigger = np.sum(np.abs(d[d < 0]))
            d = currentBigger*2 + targetBigger
            if bestDistance == None or d < bestDistance:
                bestDistance = d
    return bestDistance

def analyzeAsymmetric(a,b):
    # a = target
    # b = current
    # if you see a pixel in current that isn't in target, that's really bad
    # if you see a pixel and target that isn't an current, that's not so bad
    import cv2
    kernelSize = blurKernelSize

   
    a = cv2.GaussianBlur(a,(kernelSize,kernelSize),sigmaX = 0)
    b = cv2.GaussianBlur(b,(kernelSize,kernelSize),sigmaX = 0)

    showImage(a + b)
    
    d = a - b
    targetBigger = np.sum(d[d > 0]*d[d > 0])
    currentBigger = np.sum(d[d < 0]*d[d < 0])
    print("targetBigger = %f"%targetBigger)
    print("currentBigger = %f"%currentBigger)

#    showImage(b)

    return currentBigger*2 + targetBigger

