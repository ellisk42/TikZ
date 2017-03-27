from batch import BatchIterator
from language import *
from render import render

import matplotlib.pyplot as plot
import sys
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from time import time
import pickle
import io
import cProfile

learning_rate = 0.001

TOKENS = range(3)
STOP = TOKENS[0]
CIRCLE = TOKENS[1]
LINE = TOKENS[2]

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

def loadPrograms(filenames):
    return [ pickle.load(open(n,'rb')) for n in filenames ]

def loadExamples(numberOfExamples, filePrefix, dummyImages = True):
    programs = loadPrograms([ "%s-%d.p"%(filePrefix,j)
                              for j in range(numberOfExamples) ])
    startingExamples = []
    endingExamples = []
    target = [[],[],[],[],[]]

    startTime = time()
    # get one example from each line of each program
    for j,program in enumerate(programs):
        trace = [ "%s-%d-%d.png"%(filePrefix, j, k) for k in range(len(program)) ]
        if not dummyImages:
            trace = loadImages(trace)
        else:
            loadImages(trace) # puts them into IMAGEBYTES
        targetImage = trace[-1]
        currentImage = "blankImage" if dummyImages else np.zeros(targetImage.shape)
        for k,l in enumerate(program.lines):
            startingExamples.append(currentImage)
            endingExamples.append(targetImage)
            currentImage = trace[k]
            if isinstance(l,Circle):
                x,y = l.center.x,l.center.y
                target[0].append(CIRCLE)
                target[1].append(x)
                target[2].append(y)
                target[3].append(0)
                target[4].append(0)
            elif isinstance(l,Line):
                target[0].append(LINE)
                target[1].append(l.points[0].x)
                target[2].append(l.points[0].y)
                target[3].append(l.points[1].x)
                target[4].append(l.points[1].y)
            else:
                raise Exception('Unhandled line:'+str(l))
        # end of program
        startingExamples.append(targetImage)
        endingExamples.append(targetImage)
        target[0].append(STOP)
        target[1].append(0)
        target[2].append(0)
        target[3].append(0)
        target[4].append(0)
            
    targetVectors = [np.array(t) for t in target ]

    print "loaded images in",(time() - startTime),"s"
    
    return np.array(startingExamples), np.array(endingExamples), targetVectors

# we output 4 categorical distributions over ten choices
OUTPUTDIMENSIONS = [len(TOKENS),10,10,10,10]

class RecognitionModel():
    def __init__(self):
        # current and goal images
        self.currentPlaceholder = tf.placeholder(tf.float32, [None, 256, 256])
        self.goalPlaceholder = tf.placeholder(tf.float32, [None, 256, 256])
        # what is the target category?
        self.targetPlaceholder = [ tf.placeholder(tf.int32, [None]) for _ in OUTPUTDIMENSIONS ]
        # do we actually care about the prediction made?
        #self.targetMaskPlaceholder = [ tf.placeholder(tf.float32, [None]) for _ in OUTPUTDIMENSIONS ]

        imageInput = tf.stack([self.currentPlaceholder,self.goalPlaceholder], axis = 3)
        print "imageInput",imageInput

        numberOfFilters = [1]
        c1 = tf.layers.conv2d(inputs = imageInput,
                              filters = numberOfFilters[0],
                              kernel_size = [8,8],
                              padding = "same",
                              activation = tf.nn.relu,
                              strides = 4)
        c1 = tf.layers.max_pooling2d(inputs = c1,
                                     pool_size = 4,
                                     strides = 2,
                                     padding = "same")
        c1d = int(c1.shape[1]*c1.shape[2]*c1.shape[3])
        print c1d
        
        f1 = tf.reshape(c1, [-1, c1d])

        self.prediction = [ tf.layers.dense(f1, dimension, activation = None) for dimension in OUTPUTDIMENSIONS ]

        self.hard = [ tf.cast(tf.argmax(p,dimension = 1),tf.int32) for p in self.prediction ]

        self.averageAccuracy = reduce(tf.logical_and,
                                      [tf.equal(h,t) for h,t in zip(self.hard,self.targetPlaceholder)])
        self.averageAccuracy = tf.reduce_mean(tf.cast(self.averageAccuracy, tf.float32))

        self.loss = sum([ tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = t,logits = p))
                          for t,p in zip(self.targetPlaceholder, self.prediction) ])

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def train(self, numberOfExamples, exampleType, checkpoint = "/tmp/model.checkpoint"):
        partialImages,targetImages,targetVectors = loadExamples(numberOfExamples,
                                                                "syntheticTrainingData/"+exampleType)
        initializer = tf.global_variables_initializer()
        iterator = BatchIterator(50,tuple([partialImages,targetImages] + targetVectors),
                                 testingFraction = 0.1, stringProcessor = loadImage)
        iterator.registerPlaceholders([self.currentPlaceholder, self.goalPlaceholder] + self.targetPlaceholder)
        saver = tf.train.Saver()

        with tf.Session() as s:
            s.run(initializer)
            for i in range(100):
                _,l,accuracy = s.run([self.optimizer, self.loss, self.averageAccuracy],
                                     feed_dict = iterator.nextFeed())
                if i%50 == 0:
                    print "Iteration %d (%f passes over the data): accuracy = %f, loss = %f"%(i,float(i)*iterator.batchSize/numberOfExamples,accuracy,l)
                    print "\tTesting accuracy = %f"%(s.run(self.averageAccuracy,
                                                           feed_dict = iterator.testingFeed()))
                if i%100 == 0:
                    print "Saving checkpoint: %s" % saver.save(s, checkpoint)

    def draw(self, targetImages, checkpoint = "/tmp/model.checkpoint"):
        targetImages = [np.reshape(i,(1,300,300)) for i in loadImages(targetImages) ]
        saver = tf.train.Saver()
        with tf.Session() as s:
            saver.restore(s,checkpoint)

            for targetImage in targetImages:
                showImage(targetImage[0])

                currentImage = np.zeros(targetImage.shape)
                
                currentProgram = []

                while True:
                    feed = {self.currentPlaceholder:currentImage,
                            self.goalPlaceholder: targetImage}
                    hardDecisions = s.run(self.hard,
                                          feed_dict = feed)

                    if hardDecisions[0] == CIRCLE:
                        currentProgram.append(Circle(AbsolutePoint(hardDecisions[1], hardDecisions[2]),1))
                    elif hardDecisions[0] == LINE:
                        currentProgram.append(Line([AbsolutePoint(hardDecisions[1], hardDecisions[2]),
                                                    AbsolutePoint(hardDecisions[3], hardDecisions[4])]))
                    elif hardDecisions[1] == STOP:
                        break
                    
                    p = str(Sequence(currentProgram))
                    print p,"\n"
                    currentImage = 1.0 - render([p],yieldsPixels = True)[0]
                    currentImage = np.reshape(currentImage, targetImage.shape)
                    showImage(currentImage[0])

                    

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'test':
        RecognitionModel().draw(["syntheticTrainingData/doubleCircleLine-0-2.png"])
    else:
        cProfile.run('RecognitionModel().train(1000, "doubleCircleLine")')
