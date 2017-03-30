from batch import BatchIterator
from language import *
from render import render,animateMatrices
from utilities import *

import sys
import tensorflow as tf
import os
from time import time
import pickle
import cProfile

learning_rate = 0.001

TOKENS = range(3)
STOP = TOKENS[0]
CIRCLE = TOKENS[1]
LINE = TOKENS[2]


def loadPrograms(filenames):
    return [ pickle.load(open(n,'rb')) for n in filenames ]

def loadExamples(numberOfExamples, filePrefixes, dummyImages = True):
    print "Loading examples from these prefixes: %s"%(" ".join(filePrefixes))
    programNames = [ "syntheticTrainingData/%s-%d.p"%(filePrefix,j)
                     for j in range(numberOfExamples)
                     for filePrefix in filePrefixes ]
    programs = loadPrograms(programNames)
    startingExamples = []
    endingExamples = []
    target = [[],[],[],[],[]]

    startTime = time()
    # get one example from each line of each program
    for j,program in enumerate(programs):
        trace = [ "%s-%d.png"%(programNames[j][:-2], k) for k in range(len(program)) ]
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
OUTPUTDIMENSIONS = [len(TOKENS),8,8,8,8]

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

        # smallerInput = tf.layers.max_pooling2d(inputs = imageInput,
        #                                        pool_size = 2,
        #                                        strides = 2,
        #                                        padding = "same")

        # print "smallerInput",smallerInput

        numberOfFilters = [3]
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
        print "fully connected input dimensionality:",c1d
        
        f1 = tf.reshape(c1, [-1, c1d])

        self.prediction = [ tf.layers.dense(f1, dimension, activation = None) for dimension in OUTPUTDIMENSIONS ]

        self.hard = [ tf.cast(tf.argmax(p,dimension = 1),tf.int32) for p in self.prediction ]
        self.logSoft = [ tf.nn.log_softmax(p) for p in self.prediction ]

        self.averageAccuracy = reduce(tf.logical_and,
                                      [tf.equal(h,t) for h,t in zip(self.hard,self.targetPlaceholder)])
        self.averageAccuracy = tf.reduce_mean(tf.cast(self.averageAccuracy, tf.float32))

        self.loss = sum([ tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = t,logits = p))
                          for t,p in zip(self.targetPlaceholder, self.prediction) ])

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def train(self, numberOfExamples, exampleType, checkpoint = "/tmp/model.checkpoint"):
        partialImages,targetImages,targetVectors = loadExamples(numberOfExamples,
                                                                exampleType)
        initializer = tf.global_variables_initializer()
        iterator = BatchIterator(50,tuple([partialImages,targetImages] + targetVectors),
                                 testingFraction = 0.1, stringProcessor = loadImage)
        iterator.registerPlaceholders([self.currentPlaceholder, self.goalPlaceholder] + self.targetPlaceholder)
        saver = tf.train.Saver()

        with tf.Session() as s:
            s.run(initializer)
            for i in range(10000):
                _,l,accuracy = s.run([self.optimizer, self.loss, self.averageAccuracy],
                                     feed_dict = iterator.nextFeed())
                if i%50 == 0:
                    print "Iteration %d (%f passes over the data): accuracy = %f, loss = %f"%(i,float(i)*iterator.batchSize/numberOfExamples,accuracy,l)
                    print "\tTesting accuracy = %f"%(s.run(self.averageAccuracy,
                                                           feed_dict = iterator.testingFeed()))
                if i%100 == 0:
                    print "Saving checkpoint: %s" % saver.save(s, checkpoint)

    def beam(self, targetImage, checkpoint = "/tmp/model.checkpoint", beamSize = 10):
        totalNumberOfRenders = 0
        targetImage = loadImage(targetImage)
        targetImage = np.reshape(targetImage,(256,256))
        beam = [{'program': [],
                 'output': np.zeros(targetImage.shape),
                 'logLikelihood': 0.0}]
        # once a program is finished we wrap it up in a sequence object
        def finished(x): return isinstance(x['program'], Sequence)
        
        saver = tf.train.Saver()
        with tf.Session() as s:
            saver.restore(s,checkpoint)

            for iteration in range(7):
                feed = {self.currentPlaceholder: np.array([x['output'] for x in beam ]),
                        self.goalPlaceholder: np.array([targetImage for _ in beam ])}
                decisions = s.run(self.logSoft,
                                  feed_dict = feed)

                children = []
                for j,n in enumerate(beam):
                    if finished(n): continue
                    
                    p1 = [ (decisions[1][j][x] + decisions[2][j][y], (x,y))
                           for x in range(OUTPUTDIMENSIONS[1])
                           for y in range(OUTPUTDIMENSIONS[2]) ]
                    p2 = [ (decisions[3][j][x] + decisions[4][j][y], (x,y))
                           for x in range(OUTPUTDIMENSIONS[3])
                           for y in range(OUTPUTDIMENSIONS[4]) ]
                    command = [ (decisions[0][j][c], c)
                                for c in range(OUTPUTDIMENSIONS[0]) ]

                    lineChildren = [{'program': n['program'] + [Line.absolute(x1,y1,x2,y2)],
                                     'logLikelihood': n['logLikelihood'] + ll1 + ll2 + ll3}
                                     for (ll1,c) in command
                                    for (ll2,(x1,y1)) in p1
                                    for (ll3,(x2,y2)) in p2
                                    if c == LINE ]

                    circleChildren = [{'program': n['program'] + [Circle(AbsolutePoint(x1,y1),1)],
                                     'logLikelihood': n['logLikelihood'] + ll1 + ll2 + ll3}
                                     for (ll1,c) in command
                                    for (ll2,(x1,y1)) in p1
                                    for (ll3,(x2,y2)) in p2
                                    if c == CIRCLE and x2 == 0 and y2 == 0 ]

                    stopChildren = [{'program': Sequence(n['program']),
                                     'logLikelihood': n['logLikelihood'] + ll1}
                                     for (ll1,c) in command
                                    if c == STOP ]

                    children += (lineChildren + circleChildren + stopChildren)
                beam = sorted(children, key = lambda c: -c['logLikelihood'])[:beamSize]
                outputs = render([ str(n['program'] if finished(n) else Sequence(n['program']))
                                   for n in beam ],
                                 yieldsPixels = True)
                totalNumberOfRenders += len(beam)
                for n,o in zip(beam,outputs): n['output'] = 1.0 - o

                print "Iteration %d: %d total renders.\n"%(iteration+1,totalNumberOfRenders)
                # Show all of the finished programs
                for n in beam:
                    if finished(n):
                        print "Finished program:"
                        print n['program']
                        print "Absolute pixel-wise distance: %f"%(np.sum(np.abs(n['output'] - targetImage)))
                        print ""
                        trace = [str(Sequence(n['program'].lines[:j])) for j in range(len(n['program'])+1) ]
                        animateMatrices(render(trace,yieldsPixels = True),"neuralAnimation.gif")
                        

                    

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'test':
        RecognitionModel().beam("challenge.png",beamSize = 10)
    else:
        RecognitionModel().train(1000, ["doubleCircleLine","doubleCircle","tripleCircle","doubleLine"])
