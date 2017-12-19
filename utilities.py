import os
from random import random,shuffle,seed
import math
import sys
import io
import numpy as np
from PIL import Image


def NIPSPRIMITIVES(): return True

def log2(x):
    return math.log(x)/math.log(2)
def log10(x):
    return math.log(x)/math.log(10)

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

def loadExpert(n):
    return loadImage("drawings/expert-%d.png"%n)

def showImage(image):
    import matplotlib.pyplot as plot
    if len(image.shape) == 2:
        plot.imshow(image,cmap = 'gray')
    else:
        plot.imshow(image)
    plot.show()

def saveMatrixAsImage(m,f):
    Image.fromarray(m).convert('RGB').save(f)

def indent(s):
    return "\n".join("  "+x for x in s.splitlines())

def crossEntropyWithMask(labels, masks, predictions):
    import tensorflow as tf
    crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels,logits = predictions)
    print "crossEntropy = ",crossEntropy
    # zero out anything that is not in the mask
    masked = tf.multiply(crossEntropy,mask)
    print "masked = ",masked
    l = tf.reduce_sum(masked)
    print "l = ",l
    return l

def linesIntersect(p1,q1,p2,q2,precision = 1):
    for p in [p1,p2,q1,q2]:
        p.x = round(p.x,precision)
        p.y = round(p.y,precision)
    return linesIntersect_(p1,q1,p2,q2)    
def linesIntersect_(p1,q1,p2,q2):
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

def normalizeLogs(x):
    from scipy.misc import logsumexp
    z = logsumexp(x)
    return x - z

def isFinite(x):
    return not (math.isnan(x) or math.isinf(x))
def lse(x,y):
    if not isFinite(x): return y
    if not isFinite(y): return x
    
    if x > y:
        return x + math.log(1 + math.exp(y - x))
    else:
        return y + math.log(1 + math.exp(x - y))
def lseList(l):
    if l == []: return float('-inf')
    a = l[0]
    for x in l[1:]: a = lse(a,x)
    return a
    
def sampleLogMultinomial(logLikelihoods):
    z = lseList(logLikelihoods)
    ps = [math.exp(p - z) for p in logLikelihoods ]
    return np.random.multinomial(1,ps).tolist().index(1)

def flushEverything():
    sys.stdout.flush()
    sys.stderr.flush()


def perturbNoisyIntensities(b):
    # b: None*256*256
    p = np.copy(b)
    for j in range(b.shape[0]):
        f = random() + 0.5
        p[j,:,:] = f*b[j,:,:]
    p[p > 1] = 1.0
    return p

    
def perturbOffset(b):
    p = np.copy(b)
    w = 4
    for j in range(b.shape[0]):
        dx = int(random()*(w*2 + 1)) - w
        dy = int(random()*(w*2 + 1)) - w
        p[j,:,:] = np.roll(np.roll(p[j,:,:], dx, axis = 1), dy, axis = 0)
    return p

def perturbSaltAndPepper(p,n = 10):
    (c,w,h) = p.shape
    for j in range(c):
        for _ in range(n):
            x = int(random()*w)
            y = int(random()*h)
            p[j,x,y] = 1 - p[j,x,y]
    return p
    

def augmentData(b): return perturbSaltAndPepper(perturbOffset(perturbNoisyIntensities(b)))
    
def translateArray(a,dx,dy):
    return np.roll(np.roll(a,dx,axis = 1),dy,axis = 0)

def meanAndStandardError(x):
    mean = sum(x)/float(len(x))
    variance = sum([(z - mean)*(z - mean) for z in x ])/len(x)
    deviation = math.sqrt(variance)
    standardError = deviation/math.sqrt(len(x))
    return "%f +/- %f"%(mean,standardError)


def removeBorder(x):
    while np.all(x[0,:] < 0.1): x = x[1:,:]
    while np.all(x[x.shape[0]-1,:] < 0.1): x = x[:x.shape[0]-2,:]
    while np.all(x[:,0] < 0.1): x = x[:,1:]
    while np.all(x[:,x.shape[1]-1] < 0.1): x = x[:,:x.shape[1]-2]
    return x

def frameImageNicely(x):
    x = removeBorder(x)
    l = max([x.shape[0],x.shape[1]])
    b = 15
    z = np.zeros((l + 2*b,l + 2*b))
    z[b:x.shape[0]+b,b:x.shape[1]+b] = x


    import scipy.ndimage
    return scipy.ndimage.zoom(z,256.0/(2*b+l))

        

def mergeDictionaries(a,b):
    return dict([ (k,a.get(k,0) + b.get(k,0))
                  for k in set(a.keys() + b.keys()) ])

def picturesInDirectory(d):
    if d.endswith('.png'): return [d]
    if not d.endswith('/'): d = d + '/'
    return [ d + f for f in os.listdir(d) if f.endswith('.png') ]

def randomlyPermuteList(l):
    l = list(l)
    shuffle(l)
    return l

def fst(x): return x[0]
def snd(x): return x[1]

def flattenImageOutput(c1):
    import tensorflow as tf
    c1d = int(c1.shape[1]*c1.shape[2]*c1.shape[3])
    print "fully connected input dimensionality:",c1d

    f1 = tf.reshape(c1, [-1, c1d])
    return f1

def applyLinearTransformation(t,p):
    p = np.array([p[0],p[1],1.0])
    t = t.reshape((2,3))
    return np.matmul(t,p).tolist()

def invertTransformation(t):
    return np.linalg.inv(np.concatenate([t.reshape((2,3)),np.array([0,0,1.0]).reshape((1,3))]))[[0,1],:]
    
def integrateNormalDensity(lb,ub,mu = 0,sigma = 1):
    from scipy.stats import norm
    assert not (ub < lb)
    lessThanUpper = norm.logcdf(ub,loc = mu,scale = sigma)
    lessThanLower = norm.logcdf(lb,loc = mu,scale = sigma)
    #print lessThanUpper,lessThanLower,lessThanUpper-lessThanLower,1 - math.exp(lessThanLower - lessThanUpper)
    return lessThanUpper + np.log1p(-math.exp(lessThanLower - lessThanUpper))

def removeDuplicateStrings(xs):
    t = {}
    for x in xs: t[str(x)] = x
    return t.values()

def reflectPoint(a,r,px,py):
    if a == 'x': return (r - px,py)
    if a == 'y': return (px,r - py)
    assert False

def frequencyOfMode(l):
    if l == []: return 0
    f = {}
    for x in l: f[x] = 1 + f.get(x,0)
    return max(f.values())

def interleaveGenerators(generators):
    while generators != []:
        try:
            yield next(generators[0])
            generators = generators[1:] + [generators[0]]
        except StopIteration:
            generators = generators[1:]

def crossValidate(fullDataSet, folds = 10, randomSeed = 0, doNotPermute = False):
    if folds == 1:
        print "crossValidate: Not doing cross validation because I only have one fold."
        yield fullDataSet, fullDataSet
    else:
        n = len(fullDataSet)
        seed(randomSeed)
        if doNotPermute: indices = range(n)
        else: indices = randomlyPermuteList(range(n))
        
        smallTestSize = int(math.floor(n/float(folds)))
        bigTestSize = int(math.ceil(n/float(folds)))
        numberOfBig = n - folds*smallTestSize

        start = 0
        thingsWeHaveTestedOn = set([])
        for f in range(folds):
            if f < numberOfBig: ending = start + bigTestSize
            else: ending = start + smallTestSize
            train = indices[:start] + indices[ending:]
            test = indices[start:ending]
            start = ending
            assert len(train) + len(test) == n
            assert len(set(train)&set(test)) == 0
            assert len(set(test)&thingsWeHaveTestedOn) == 0
            thingsWeHaveTestedOn = thingsWeHaveTestedOn|set(test)
            if f == folds - 1:
                assert len(thingsWeHaveTestedOn) == n
            yield ([ fullDataSet[j] for j in train ], [ fullDataSet[j] for j in test ])
def parallelMap(numberOfCPUs, f, *xs):
    from pathos.multiprocessing import ProcessingPool as Pool

    numberOfCPUs = min(numberOfCPUs,len(zip(*xs)))
    
    if numberOfCPUs == 1: return map(f,*xs)
    def safeCall(x):
        try:
            y = f(*x)
            return y
        except Exception as e:
            print "Exception in worker during parallel map:\n%s"%(traceback.format_exc())
            raise e
    return Pool(numberOfCPUs).map(safeCall,zip(*xs))

if __name__ == "__main__":
    for x in interleaveGenerators([iter(range(9)),iter(xrange(3))]):
        print x
    assert False
    for train, test in crossValidate(range(98),20,doNotPermute = True):
        print train, test
