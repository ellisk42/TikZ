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
    target = {}

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
            for j,t in enumerate(PrimitiveDecoder.extractTargets(l)):
                target[j] = target.get(j,[]) + [t]
        # end of program
        startingExamples.append(targetImage)
        endingExamples.append(targetImage)
        for j in target:
            target[j] += [STOP] # should be zero and therefore valid for everyone
            
    targetVectors = [np.array(target[j]) for j in sorted(target.keys()) ]

    print "loaded images in",(time() - startTime),"s"
    print "target dimensionality:",len(targetVectors)
    
    return np.array(startingExamples), np.array(endingExamples), targetVectors

class StandardPrimitiveDecoder():
    def makeNetwork(self,imageRepresentation):
        self.prediction = []
        self.targetPlaceholder = [ tf.placeholder(tf.int32, [None]) for _ in self.outputDimensions ]
        predictionInputs = imageRepresentation
        for j,d in enumerate(self.outputDimensions):
            self.prediction.append(tf.layers.dense(predictionInputs, d, activation = None))
            predictionInputs = tf.concat([predictionInputs,
                                          tf.one_hot(self.targetPlaceholder[j], d)],
                                         axis = 1)
        self.hard = [ tf.cast(tf.argmax(p,dimension = 1),tf.int32) for p in self.prediction ]
        self.soft = [ tf.nn.log_softmax(p) for p in self.prediction ]

    def loss(self):
        return sum([ tf.nn.sparse_softmax_cross_entropy_with_logits(labels = l, logits = p)
                     for l,p in zip(self.targetPlaceholder, self.prediction) ])
    def accuracyVector(self):
        return reduce(tf.logical_and,
                      [tf.equal(h,t) for h,t in zip(self.hard,self.targetPlaceholder)])
    def placeholders(self): return self.targetPlaceholder

    def beamTrace(self, session, feed, beamSize):
        originalFeed = feed
        feed = dict([(k,feed[k]) for k in feed])
        
        traces = [(0.0,[])]
        for j in range(len(self.outputDimensions)):
            for k in range(j):
                feed[self.targetPlaceholder[k]] = np.array([ t[1][k] for t in traces ])
            for p in originalFeed:
                feed[p] = np.repeat(originalFeed[p], len(traces), axis = 0)
            soft = session.run(self.soft[j], feed_dict = feed)
            traces = [(s + coordinateScore, trace + [coordinateIndex])
                  for traceIndex,(s,trace) in enumerate(traces)
                  for coordinateIndex,coordinateScore in enumerate(soft[traceIndex]) ]
            traces = sorted(traces, key = lambda t: -t[0])[:beamSize]
        return traces
            


class CircleDecoder(StandardPrimitiveDecoder):
    def __init__(self, imageRepresentation):
        self.outputDimensions = [8,8] # x,y
        self.makeNetwork(imageRepresentation)
            
    def token(self): return CIRCLE

    def beam(self, session, feed, beamSize):
        return [(s, Circle(AbsolutePoint(Number(x),Number(y)),Number(1)))
                for s,[x,y] in self.beamTrace(session, feed, beamSize) ]

    @staticmethod
    def extractTargets(l):
        if isinstance(l,Circle): return [l.center.x,l.center.y]
        return [0,0]

class LineDecoder(StandardPrimitiveDecoder):
    def __init__(self, imageRepresentation):
        self.outputDimensions = [8,8,8,8,2,2] # x,y for beginning and end; arrow/-
        self.makeNetwork(imageRepresentation)

    def token(self): return LINE
    def beam(self, session, feed, beamSize):
        # def enumerateTraces(j):
        #     traces = []
        #     if j == len(self.outputDimensions): return [(0,[])]
        #     distribution = session.run(self.soft[j], feed_dict = feed)[0]
        #     for x,s in enumerate(distribution):
        #         feed[self.targetPlaceholder[j]] = np.array([x])
        #         for suffixScore,suffix in enumerateTraces(j+1):
        #             traces.append((s + suffixScore, [x]+suffix))
        #     del feed[self.targetPlaceholder[j]]
        #     return traces

        return [(s, Line.absolute(Number(x1),Number(y1),Number(x2),Number(y2),arrow = arrow,solid = solid))
                for s,[x1,y1,x2,y2,arrow,solid] in self.beamTrace(session, feed, beamSize) ]

    @staticmethod
    def extractTargets(l):
        if isinstance(l,Line):
            return [l.points[0].x,
                    l.points[0].y,
                    l.points[1].x,
                    l.points[1].y,
                    int(l.arrow),
                    int(l.solid)]
        return [0]*6

class StopDecoder():
    def __init__(self, imageRepresentation):
        self.outputDimensions = []
    def loss(self): return 0.0
    def token(self): return STOP
    def placeholders(self): return []
    def softPredictions(self): return []

class PrimitiveDecoder():
    def __init__(self, imageRepresentation):
        self.decoders = [CircleDecoder(imageRepresentation),
                         LineDecoder(imageRepresentation),
                         StopDecoder(imageRepresentation)]

        self.prediction = tf.layers.dense(imageRepresentation, len(self.decoders))
        self.hard = tf.cast(tf.argmax(self.prediction,dimension = 1),tf.int32)
        self.soft = tf.nn.log_softmax(self.prediction)
        self.targetPlaceholder = tf.placeholder(tf.int32, [None])

    def loss(self):
        # the first label is for the primitive category
        ll = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.targetPlaceholder,
                                                                          logits = self.prediction))
        for decoder in self.decoders:
            decoderLosses = decoder.loss()
            decoderMask = tf.cast(tf.equal(self.targetPlaceholder, decoder.token()), tf.float32)
            decoderLoss = tf.reduce_sum(tf.multiply(decoderMask,decoderLosses))
            ll += decoderLoss

        return ll

    def accuracy(self):
        a = tf.equal(self.hard,self.targetPlaceholder)
        for decoder in self.decoders:
            if decoder.token() != STOP:
                a = tf.logical_and(a,
                                   tf.logical_or(decoder.accuracyVector(),
                                                 tf.not_equal(self.hard,decoder.token())))
        return tf.reduce_mean(tf.cast(a, tf.float32))

    def placeholders(self):
        p = [self.targetPlaceholder]
        for d in self.decoders: p += d.placeholders()
        return p

    @staticmethod
    def extractTargets(l):
        t = [STOP]
        if isinstance(l,Circle): t = [CIRCLE]
        if isinstance(l,Line): t = [LINE]
        return t + CircleDecoder.extractTargets(l) + LineDecoder.extractTargets(l)

    def beam(self, session, feed, beamSize):
        tokenScores = session.run(self.soft, feed_dict = feed)[0]
        b = [(tokenScores[STOP], None)] # STOP
        for d in self.decoders:
            if d.token() == STOP: continue
            b += [ (s + tokenScores[d.token()], program)
                   for (s, program) in d.beam(session, feed, beamSize) ]
        return b

class RecognitionModel():
    def __init__(self):
        # current and goal images
        self.currentPlaceholder = tf.placeholder(tf.float32, [None, 256, 256])
        self.goalPlaceholder = tf.placeholder(tf.float32, [None, 256, 256])

        imageInput = tf.stack([self.currentPlaceholder,self.goalPlaceholder], axis = 3)

        numberOfFilters = [5]
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

        self.decoder = PrimitiveDecoder(f1)
        self.loss = self.decoder.loss()
        self.averageAccuracy = self.decoder.accuracy()

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)        


    def train(self, numberOfExamples, exampleType, checkpoint = "/tmp/model.checkpoint"):
        partialImages,targetImages,targetVectors = loadExamples(numberOfExamples,
                                                                exampleType)
        initializer = tf.global_variables_initializer()
        iterator = BatchIterator(50,tuple([partialImages,targetImages] + targetVectors),
                                 testingFraction = 0.05, stringProcessor = loadImage)
        iterator.registerPlaceholders([self.currentPlaceholder, self.goalPlaceholder] +
                                      self.decoder.placeholders())
        saver = tf.train.Saver()

        with tf.Session() as s:
            s.run(initializer)
            for e in range(100):
                epicLoss = []
                epicAccuracy = []
                for feed in iterator.epochFeeds():
                    _,l,accuracy = s.run([self.optimizer, self.loss, self.averageAccuracy],
                                         feed_dict = feed)
                    epicLoss.append(l)
                    epicAccuracy.append(accuracy)
                print "Epoch %d: accuracy = %f, loss = %f"%((e+1),sum(epicAccuracy)/len(epicAccuracy),sum(epicLoss)/len(epicLoss))
                print "\tTesting accuracy = %f"%(s.run(self.averageAccuracy,
                                                       feed_dict = iterator.testingFeed()))
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
                children = []
                for parent in beam:
                    feed = {self.currentPlaceholder: np.array([parent['output']]),
                            self.goalPlaceholder: np.array([targetImage])}

                    
                    for childScore,suffix in self.decoder.beam(s, feed, beamSize):
                        if suffix == None:
                            k = Sequence(parent['program'])
                        else:
                            k = parent['program'] + [suffix]
                        children.append({'program': k,
                                         'logLikelihood': parent['logLikelihood'] + childScore})
                
                beam = sorted(children, key = lambda c: -c['logLikelihood'])[:beamSize]
                outputs = render([ (n['program'] if finished(n) else Sequence(n['program'])).TikZ()
                                   for n in beam ],
                                 yieldsPixels = True)
                totalNumberOfRenders += len(beam)
                for n,o in zip(beam,outputs): n['output'] = 1.0 - o

                print "Iteration %d: %d total renders.\n"%(iteration+1,totalNumberOfRenders)
                # Show all of the finished programs
                for n in beam:
                    if finished(n):
                        print "Finished program:"
                        print n['program'].TikZ()
                        print "Absolute pixel-wise distance: %f"%(np.sum(np.abs(n['output'] - targetImage)))
                        print ""
                        trace = [Sequence(n['program'].lines[:j]).TikZ() for j in range(len(n['program'])+1) ]
                        animateMatrices(render(trace,yieldsPixels = True),"neuralAnimation.gif")
                        

                    

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'test':
        RecognitionModel().beam("challenge.png",beamSize = 10)
    else:
        RecognitionModel().train(1000, ["doubleCircleLine","doubleCircle","tripleCircle","doubleLine","individualCircle"])
