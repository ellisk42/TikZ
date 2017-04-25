import sys
import io
import numpy as np
from PIL import Image
import tensorflow as tf

def image2array(p):
    p = p.convert('L')
    (w,h) = p.size
    return 1.0 - np.array(p,np.uint8).reshape((h,w))/255.0

# maps from a file named to the bytes in that file
IMAGEBYTES = {}
def loadImage(n):
    if n == "blankImage": return np.zeros((256,256))
    if not n in IMAGEBYTES:
        with open(n,'rb') as handle:
            IMAGEBYTES[n] = handle.read()
    p = Image.open(io.BytesIO(IMAGEBYTES[n]))
    # most of the time is spent in here for some reason
    return image2array(p)
def loadImages(ns): return map(lambda n: loadImage(n),ns)

def cacheImage(n,content): IMAGEBYTES[n] = content

def showImage(image):
    import matplotlib.pyplot as plot
    plot.imshow(image,cmap = 'gray')
    plot.show()

def saveMatrixAsImage(m,f):
    Image.fromarray(m).convert('RGB').save(f)


def crossEntropyWithMask(labels, masks, predictions):
    crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels,logits = predictions)
    print "crossEntropy = ",crossEntropy
    # zero out anything that is not in the mask
    masked = tf.multiply(crossEntropy,mask)
    print "masked = ",masked
    l = tf.reduce_sum(masked)
    print "l = ",l
    return l
    
def linesIntersect(p1,q1,p2,q2):
    def onSegment(p,q,r):
        return q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y)
    def orientation(p,q,r):
        val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
        if val == 0: return 0 # colinear
        if val > 0: return 1
        return 2

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    # Special Cases
    # p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if (o1 == 0 and onSegment(p1, p2, q1)): return True
 
    # p1, q1 and p2 are colinear and q2 lies on segment p1q1
    if (o2 == 0 and onSegment(p1, q2, q1)): return True
 
    # p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if (o3 == 0 and onSegment(p2, p1, q2)): return True
 
    # p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if (o4 == 0 and onSegment(p2, q1, q2)): return True
 
    return False

def truncatedNormal(lower = None,upper = None):
    x = np.random.normal()
    if lower != None and x < lower: return truncatedNormal(lower = lower,upper = upper)
    if upper != None and x > upper: return truncatedNormal(lower = lower,upper = upper)
    return x
def flushEverything():
    sys.stdout.flush()
    sys.stderr.flush()
