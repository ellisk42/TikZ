import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plot
import tensorflow as tf

IMAGEBYTES = {}
def loadImage(n):
    if n == "blankImage": return np.zeros((256,256))
    def processPicture(p):
        # most of the time is spent in here for some reason
        p = p.convert('L')
        (w,h) = p.size
        return 1.0 - np.array(p,np.uint8).reshape((h,w))/255.0
    if not n in IMAGEBYTES:
        with open(n,'rb') as handle:
            IMAGEBYTES[n] = handle.read()
    return processPicture(Image.open(io.BytesIO(IMAGEBYTES[n])))
def loadImages(ns): return map(loadImage,ns)

def showImage(image):
    plot.imshow(image,cmap = 'gray')
    plot.show()


def crossEntropyWithMask(labels, masks, predictions):
    crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels,logits = predictions)
    print "crossEntropy = ",crossEntropy
    # zero out anything that is not in the mask
    masked = tf.multiply(crossEntropy,mask)
    print "masked = ",masked
    l = tf.reduce_sum(masked)
    print "l = ",l
    return l
    
