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

# The data is generated on a  MAXIMUMCOORDINATExMAXIMUMCOORDINATE grid
# We can interpolate between stochastic search and neural networks by downsampling to a smaller grid
APPROXIMATINGGRID = MAXIMUMCOORDINATE

learning_rate = 0.001

[STOP,CIRCLE,LINE,RECTANGLE] = range(4)


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
        noisyTarget = "%s-noisy.png"%(programNames[j][:-2])
        if not dummyImages:
            trace = loadImages(trace)
            noisyTarget = loadImage(noisyTarget)
        else:
            loadImages(trace + [noisyTarget]) # puts them into IMAGEBYTES
        targetImage = trace[-1]
        currentImage = "blankImage" if dummyImages else np.zeros(targetImage.shape)
        for k,l in enumerate(program.lines):
            startingExamples.append(currentImage)
            endingExamples.append(noisyTarget)
            currentImage = trace[k]
            for j,t in enumerate(PrimitiveDecoder.extractTargets(l)):
                target[j] = target.get(j,[]) + [t]
        # end of program
        startingExamples.append(targetImage)
        endingExamples.append(noisyTarget)
        for j in target:
            target[j] += [STOP] # should be zero and therefore valid for everyone
            
    targetVectors = [np.array(target[j]) for j in sorted(target.keys()) ]

    print "loaded images in",(time() - startTime),"s"
    print "target dimensionality:",len(targetVectors)
    
    return np.array(startingExamples), np.array(endingExamples), targetVectors

class StandardPrimitiveDecoder():
    def makeNetwork(self,imageRepresentation):
        # A placeholder for each target
        self.targetPlaceholder = [ tf.placeholder(tf.int32, [None]) for _ in self.outputDimensions ]

        # A prediction for each target
        self.prediction = []
        # populate self.production
        predictionInputs = imageRepresentation
        for j,d in enumerate(self.outputDimensions):
            self.prediction.append(tf.layers.dense(predictionInputs, d, activation = None))
            predictionInputs = tf.concat([predictionInputs,
                                          tf.one_hot(self.targetPlaceholder[j], d)],
                                         axis = 1)
        # "hard" predictions (integers)
        self.hard = [ tf.cast(tf.argmax(p,dimension = 1),tf.int32) for p in self.prediction ]

        # "soft" predictions (logits)
        self.soft = [ tf.nn.log_softmax(p) for p in self.prediction ]

    def loss(self):
        return sum([ tf.nn.sparse_softmax_cross_entropy_with_logits(labels = l, logits = p)
                     for l,p in zip(self.targetPlaceholder, self.prediction) ])
    def accuracyVector(self):
        '''For each example in the batch, do hard predictions match the target? ty = [None,bool]'''
        return reduce(tf.logical_and,
                      [tf.equal(h,t) for h,t in zip(self.hard,self.targetPlaceholder)])
    def placeholders(self): return self.targetPlaceholder

    @property
    def token(self): return self.__class__.token

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
    token = CIRCLE
    languagePrimitive = Circle
    
    def __init__(self, imageRepresentation):
        self.outputDimensions = [APPROXIMATINGGRID,APPROXIMATINGGRID] # x,y
        self.makeNetwork(imageRepresentation)
    
    def beam(self, session, feed, beamSize):
        return [(s, Circle(AbsolutePoint(Number(x),Number(y)),Number(1)))
                for s,[x,y] in self.beamTrace(session, feed, beamSize) ]

    @staticmethod
    def extractTargets(l):
        if isinstance(l,Circle): return [l.center.x.n, l.center.y.n]
        return [0,0]

class RectangleDecoder(StandardPrimitiveDecoder):
    token = RECTANGLE
    languagePrimitive = Rectangle

    def __init__(self, imageRepresentation):
        self.outputDimensions = [APPROXIMATINGGRID,APPROXIMATINGGRID,APPROXIMATINGGRID,APPROXIMATINGGRID] # x,y
        self.makeNetwork(imageRepresentation)
            

    def beam(self, session, feed, beamSize):
        return [(s, Rectangle(AbsolutePoint(Number(x1),Number(y1)),
                              AbsolutePoint(Number(x2),Number(y2))))
                for s,[x1,y1,x2,y2] in self.beamTrace(session, feed, beamSize) ]

    @staticmethod
    def extractTargets(l):
        if isinstance(l,Rectangle): return [l.p1.x.n,l.p1.y.n,l.p2.x.n,l.p2.y.n]
        return [0]*4

class LineDecoder(StandardPrimitiveDecoder):
    token = LINE
    languagePrimitive = Line

    def __init__(self, imageRepresentation):
        self.outputDimensions = [APPROXIMATINGGRID,APPROXIMATINGGRID,APPROXIMATINGGRID,APPROXIMATINGGRID,2,2] # x,y for beginning and end; arrow/-
        self.makeNetwork(imageRepresentation)
    
    def beam(self, session, feed, beamSize):
        return [(s, Line.absolute(Number(x1),Number(y1),Number(x2),Number(y2),arrow = arrow,solid = solid))
                for s,[x1,y1,x2,y2,arrow,solid] in self.beamTrace(session, feed, beamSize) ]

    @staticmethod
    def extractTargets(l):
        if isinstance(l,Line):
            return [l.points[0].x.n,
                    l.points[0].y.n,
                    l.points[1].x.n,
                    l.points[1].y.n,
                    int(l.arrow),
                    int(l.solid)]
        return [0]*6

class StopDecoder():
    def __init__(self, imageRepresentation):
        self.outputDimensions = []
    def loss(self): return 0.0
    token = STOP
    languagePrimitive = None
    def placeholders(self): return []
    def softPredictions(self): return []
    @staticmethod
    def extractTargets(_): return []

class PrimitiveDecoder():
    decoderClasses = [CircleDecoder,
                      RectangleDecoder,
                      LineDecoder,
                      StopDecoder]
    def __init__(self, imageRepresentation):
        self.decoders = [k(imageRepresentation) for k in PrimitiveDecoder.decoderClasses]

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
            decoderMask = tf.cast(tf.equal(self.targetPlaceholder, decoder.token), tf.float32)
            decoderLoss = tf.reduce_sum(tf.multiply(decoderMask,decoderLosses))
            ll += decoderLoss

        return ll

    def accuracy(self):
        a = tf.equal(self.hard,self.targetPlaceholder)
        for decoder in self.decoders:
            if decoder.token != STOP:
                a = tf.logical_and(a,
                                   tf.logical_or(decoder.accuracyVector(),
                                                 tf.not_equal(self.hard,decoder.token)))
        return tf.reduce_mean(tf.cast(a, tf.float32))

    def placeholders(self):
        p = [self.targetPlaceholder]
        for d in self.decoders: p += d.placeholders()
        return p

    @staticmethod
    def extractTargets(l):
        '''Given a line of code l, what is the array of targets (int's) we expect the decoder to produce?'''
        t = [STOP]
        for d in PrimitiveDecoder.decoderClasses:
            if isinstance(l,d.languagePrimitive):
                t = [d.token]
                break
        for d in PrimitiveDecoder.decoderClasses:
            t += d.extractTargets(l)
        return t

    def beam(self, session, feed, beamSize):
        tokenScores = session.run(self.soft, feed_dict = feed)[0]
        b = [(tokenScores[STOP], None)] # STOP
        for d in self.decoders:
            if d.token == STOP: continue
            b += [ (s + tokenScores[d.token], program)
                   for (s, program) in d.beam(session, feed, beamSize) ]
        return b

class RecognitionModel():
    def __init__(self):
        # current and goal images
        self.currentPlaceholder = tf.placeholder(tf.float32, [None, 256, 256])
        self.goalPlaceholder = tf.placeholder(tf.float32, [None, 256, 256])

        imageInput = tf.stack([self.currentPlaceholder,self.goalPlaceholder], axis = 3)

        numberOfFilters = [8,6]
        kernelSizes = [8,8]
        
        poolSizes = [8,2]
        poolStrides = [4,2]
        nextInput = imageInput
        for filterCount,kernelSize,poolSize,poolStride in zip(numberOfFilters,kernelSizes,poolSizes,poolStrides):
            c1 = tf.layers.conv2d(inputs = nextInput,
                                  filters = filterCount,
                                  kernel_size = [kernelSize,kernelSize],
                                  padding = "same",
                                  activation = tf.nn.relu,
                                  strides = 1)
            c1 = tf.layers.max_pooling2d(inputs = c1,
                                         pool_size = poolSize,
                                         strides = poolStride,
                                         padding = "same")
            print "Convolution output:",c1
            nextInput = c1
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
                                 testingFraction = 0.0, stringProcessor = loadImage)
        iterator.registerPlaceholders([self.currentPlaceholder, self.goalPlaceholder] +
                                      self.decoder.placeholders())
        saver = tf.train.Saver()

        with tf.Session() as s:
            s.run(initializer)
            for e in range(20):
                epicLoss = []
                epicAccuracy = []
                for feed in iterator.epochFeeds():
                    _,l,accuracy = s.run([self.optimizer, self.loss, self.averageAccuracy],
                                         feed_dict = feed)
                    epicLoss.append(l)
                    epicAccuracy.append(accuracy)
                print "Epoch %d: accuracy = %f, loss = %f"%((e+1),sum(epicAccuracy)/len(epicAccuracy),sum(epicLoss)/len(epicLoss))
                testingAccuracy = [ s.run(self.averageAccuracy, feed_dict = feed) for feed in iterator.testingFeeds() ]
                print "\tTesting accuracy = %f"%(sum(testingAccuracy)/len(testingAccuracy))
                print "Saving checkpoint: %s" % saver.save(s, checkpoint)

    def analyzeFailures(self, numberOfExamples, exampleType, checkpoint):
        partialImages,targetImages,targetVectors = loadExamples(numberOfExamples,
                                                                exampleType)
        iterator = BatchIterator(1,tuple([partialImages,targetImages] + targetVectors),
                                 testingFraction = 0.0, stringProcessor = loadImage)
        iterator.registerPlaceholders([self.currentPlaceholder, self.goalPlaceholder] +
                                      self.decoder.placeholders())
        saver = tf.train.Saver()
        failureLog = [] # pair of current goal

        with tf.Session() as s:
            saver.restore(s,checkpoint)
            for feed in iterator.epochFeeds():
                accuracy = s.run(self.averageAccuracy,
                                 feed_dict = feed)
                assert accuracy == 0.0 or accuracy == 1.0
                if accuracy < 0.5:
                    # decode the action preferred by the model
                    preferredLine = max(self.decoder.beam(s, {self.currentPlaceholder: feed[self.currentPlaceholder],
                                                              self.goalPlaceholder: feed[self.goalPlaceholder]}, 1),
                                        key = lambda foo: foo[0])[1]
                    preferredLine = "\n%end of program\n" if preferredLine == None else preferredLine.TikZ()
                    failureLog.append((feed[self.currentPlaceholder][0], feed[self.goalPlaceholder][0], preferredLine))
                    

        print "Failures:",len(failureLog),'/',iterator.trainingSetSize
        for j,(c,g,l) in enumerate(failureLog):
            saveMatrixAsImage(c*255,"failures/%d-current.png"%j)
            saveMatrixAsImage(g*255,"failures/%d-goal.png"%j)
            pixels = render([l],yieldsPixels = True,canvas = (MAXIMUMCOORDINATE,MAXIMUMCOORDINATE))[0]
            pixels = 1.0 - pixels
            saveMatrixAsImage(pixels*255 + 255*c,"failures/%d-predicted.png"%j)
                

    def beam(self, targetImage, checkpoint = "/tmp/model.checkpoint", beamSize = 10):
        totalNumberOfRenders = 0
        targetImage = loadImage(targetImage)
        showImage(targetImage)
        targetImage = np.reshape(targetImage,(256,256))
        beam = [{'program': [],
                 'output': np.zeros(targetImage.shape),
                 'logLikelihood': 0.0}]
        # once a program is finished we wrap it up in a sequence object
        def finished(x): return isinstance(x['program'], Sequence)

        finishedPrograms = []
        
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
                # record all of the finished programs
                finishedPrograms += [ n for n in beam if finished(n) ]
                # Remove all of the finished programs
                beam = [ n for n in beam if not finished(n) ]
                if beam == []:
                    print "Empty beam."
                    break

            print "Finished programs, sorted by likelihood:"
            finishedPrograms.sort(key = lambda n: -n['logLikelihood'])
            for n in finishedPrograms[:3]:
                print "Finished program: log likelihood %f"%(n['logLikelihood'])
                print n['program'].TikZ()
                print "Absolute pixel-wise distance: %f"%(np.sum(np.abs(n['output'] - targetImage)))
                print ""
                trace = [Sequence(n['program'].lines[:j]).TikZ() for j in range(len(n['program'])+1) ]
                animateMatrices(render(trace,yieldsPixels = True),"neuralAnimation.gif")            

                        

                    

if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'test':
        RecognitionModel().beam(sys.argv[2],
                                beamSize = 20,
                                checkpoint = "checkpoints/model.checkpoint")
    elif sys.argv[1] == 'analyze':
        RecognitionModel().analyzeFailures(100, ["randomScene"], checkpoint = "checkpoints/model.checkpoint")
    elif sys.argv[1] == 'train':
        RecognitionModel().train(10000, ["randomScene"], checkpoint = "checkpoints/model.checkpoint")
