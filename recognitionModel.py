from batch import BatchIterator
from language import *
from render import render,animateMatrices
from utilities import *
from distanceMetrics import blurredDistance,asymmetricBlurredDistance,analyzeAsymmetric
from fastRender import fastRender,loadPrecomputedRenderings


import argparse
import tarfile
import sys
import tensorflow as tf
import os
import io
from time import time
import pickle
import cProfile
from multiprocessing import Pool

# The data is generated on a  MAXIMUMCOORDINATExMAXIMUMCOORDINATE grid
# We can interpolate between stochastic search and neural networks by downsampling to a smaller grid
APPROXIMATINGGRID = MAXIMUMCOORDINATE
def coordinate2grid(c): return c*MAXIMUMCOORDINATE/APPROXIMATINGGRID
def grid2coordinate(g): return g*APPROXIMATINGGRID/MAXIMUMCOORDINATE

learning_rate = 0.001
TESTINGFRACTION = 0.1

[STOP,CIRCLE,LINE,RECTANGLE] = range(4)


def loadExamples(numberOfExamples, dummyImages = True, noisy = False):
    noisyTrainingData = noisy
    
    if os.path.isfile('/om/user/ellisk/syntheticTrainingData.tar'):
        handle = '/om/user/ellisk/syntheticTrainingData.tar'
    else:
        handle = 'syntheticTrainingData.tar'
    print "Loading data from",handle
    handle = tarfile.open(handle)
    
    # just load everything into RAM - faster that way. screw you tar
    members = {}
    for member in handle:
        if member.name == '.': continue
        stuff = handle.extractfile(member)
        members[member.name] = stuff.read()
        stuff.close()
    handle.close()

    print "Loaded tar file into RAM: %d entries."%len(members)
    
    programNames = [ "./randomScene-%d.p"%(j)
                     for j in range(numberOfExamples) ]
    programs = [ pickle.load(io.BytesIO(members[n])) for n in programNames ]

    print "Loaded pickles."

    startingExamples = []
    endingExamples = []
    target = {}

    # for debugging purposes / analysis, keep track of the target line
    targetLine = []

    startTime = time()
    # get one example from each line of each program
    for j,program in enumerate(programs):
        if j%10000 == 1:
            print "Loaded %d/%d programs"%(j - 1,len(programs))
        trace = [ "./randomScene-%d-%d.png"%(j, k) for k in range(len(program)) ]
        noisyTarget = "./randomScene-%d-noisy.png"%(j) if noisyTrainingData else trace[-1]
        # cache the images
        for imageFilename in [noisyTarget] + trace:
            cacheImage(imageFilename, members[imageFilename])
        if not dummyImages:
            trace = loadImages(trace)
            noisyTarget = loadImage(noisyTarget)
        
        targetImage = trace[-1]
        currentImage = "blankImage" if dummyImages else np.zeros(targetImage.shape)
        for k,l in enumerate(program.lines):
            startingExamples.append(currentImage)
            endingExamples.append(noisyTarget)
            targetLine.append(l)
            currentImage = trace[k]
            for j,t in enumerate(PrimitiveDecoder.extractTargets(l)):
                if not j in target: target[j] = []
                target[j].append(t)
        # end of program
        startingExamples.append(targetImage)
        endingExamples.append(noisyTarget)
        targetLine.append(None)
        for j in target:
            target[j] += [STOP] # should be zero and therefore valid for everyone
            
    targetVectors = [np.array(target[j]) for j in sorted(target.keys()) ]

    print "loaded images in",(time() - startTime),"s"
    print "target dimensionality:",len(targetVectors)

    return np.array(startingExamples), np.array(endingExamples), targetVectors, np.array(targetLine)

class StandardPrimitiveDecoder():
    def makeNetwork(self,imageRepresentation):
        # A placeholder for each target
        self.targetPlaceholder = [ tf.placeholder(tf.int32, [None]) for _ in self.outputDimensions ]
        if not hasattr(self, 'hiddenSizes'):
            self.hiddenSizes = [None]*len(self.outputDimensions)

        # A prediction for each target
        self.prediction = []
        # populate self.production
        predictionInputs = imageRepresentation
        for j,d in enumerate(self.outputDimensions):
            if self.hiddenSizes[j] == None or self.hiddenSizes[j] == 0:
                self.prediction.append(tf.layers.dense(predictionInputs, d, activation = None))
            else:
                intermediateRepresentation = tf.layers.dense(predictionInputs,
                                                             self.hiddenSizes[j],
                                                             activation = tf.nn.sigmoid)
                self.prediction.append(tf.layers.dense(intermediateRepresentation, d, activation = None))
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
        return [(s, Circle(AbsolutePoint(Number(grid2coordinate(x)),Number(grid2coordinate(y))),Number(1)))
                for s,[x,y] in self.beamTrace(session, feed, beamSize)
                if x > 1 and y > 1 and x < MAXIMUMCOORDINATE - 1 and y < MAXIMUMCOORDINATE - 1]

    @staticmethod
    def extractTargets(l):
        if isinstance(l,Circle):
            return [coordinate2grid(l.center.x.n),
                    coordinate2grid(l.center.y.n)]
        return [0,0]

class RectangleDecoder(StandardPrimitiveDecoder):
    token = RECTANGLE
    languagePrimitive = Rectangle

    def __init__(self, imageRepresentation):
        self.outputDimensions = [APPROXIMATINGGRID,APPROXIMATINGGRID,APPROXIMATINGGRID,APPROXIMATINGGRID] # x,y
        self.makeNetwork(imageRepresentation)
            

    def beam(self, session, feed, beamSize):
        return [(s, Rectangle.absolute(grid2coordinate(x1),
                                       grid2coordinate(y1),
                                       grid2coordinate(x2),
                                       grid2coordinate(y2)))
                for s,[x1,y1,x2,y2] in self.beamTrace(session, feed, beamSize)
                if x1 != x2 and y1 != y2 and x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0 ]

    @staticmethod
    def extractTargets(l):
        if isinstance(l,Rectangle):
            return [coordinate2grid(l.p1.x.n),
                    coordinate2grid(l.p1.y.n),
                    coordinate2grid(l.p2.x.n),
                    coordinate2grid(l.p2.y.n)]
        return [0]*4

class LineDecoder(StandardPrimitiveDecoder):
    token = LINE
    languagePrimitive = Line

    def __init__(self, imageRepresentation):
        self.outputDimensions = [APPROXIMATINGGRID,APPROXIMATINGGRID,APPROXIMATINGGRID,APPROXIMATINGGRID,2,2] # x,y for beginning and end; arrow/-
        self.hiddenSizes = [None,
                            32,
                            32,
                            32,
                            None,
                            None]
        self.makeNetwork(imageRepresentation)
    
    def beam(self, session, feed, beamSize):
        return [(s, Line.absolute(Number(grid2coordinate(x1)),
                                  Number(grid2coordinate(y1)),
                                  Number(grid2coordinate(x2)),
                                  Number(grid2coordinate(y2)),
                                  arrow = arrow == 1,solid = solid == 1))
                for s,[x1,y1,x2,y2,arrow,solid] in self.beamTrace(session, feed, beamSize)
                if (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) > 0 and x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0 ]

    @staticmethod
    def extractTargets(l):
        if isinstance(l,Line):
            return [coordinate2grid(l.points[0].x.n),
                    coordinate2grid(l.points[0].y.n),
                    coordinate2grid(l.points[1].x.n),
                    coordinate2grid(l.points[1].y.n),
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
    # It might matter in which order these classes are listed.
    # Because you predict circle targets, then rectangle targets, then line targets
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
        self.imageRepresentation = imageRepresentation

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
        # to accelerate beam decoding, we can cash the image representation
        [tokenScores,imageRepresentation] = session.run([self.soft,self.imageRepresentation], feed_dict = feed)
        tokenScores = tokenScores[0]
        # print "token scores ",
        # for s in tokenScores: print s," "
        # print "\nToken rectangle score: %f"%tokenScores[RectangleDecoder.token]
        feed[self.imageRepresentation] = imageRepresentation
        
        b = [(tokenScores[STOP], None)] # STOP
        for d in self.decoders:
            if d.token == STOP: continue
            b += [ (s + tokenScores[d.token], program)
                   for (s, program) in d.beam(session, feed, beamSize) ]
        # for s,p in b:
        #     print s,p
#        assert False
        return b

# Particle in sequential Monte Carlo
class Particle():
    def __init__(self, program = None,
                 time = 0.0,
                 parent = None,
                 output = None,
                 distance = None,
                 count = None,
                 logLikelihood = None,
                 score = None):
        self.time = time
        self.score = score
        self.count = count
        self.program = program
        self.parent = parent
        self.output = output
        self.distance = distance
        self.logLikelihood = logLikelihood
    # once a program is finished we wrap it up in a sedquence object
    def finished(self): return isinstance(self.program, Sequence)

class RecognitionModel():
    def __init__(self, arguments):
        self.noisy = arguments.noisy
        self.arguments = arguments
        # current and goal images
        self.currentPlaceholder = tf.placeholder(tf.float32, [None, 256, 256])
        self.goalPlaceholder = tf.placeholder(tf.float32, [None, 256, 256])

        imageInput = tf.stack([self.currentPlaceholder,self.goalPlaceholder], axis = 3)

        initialDilation = 1
        horizontalKernels = tf.layers.conv2d(inputs = imageInput,
                                             filters = 4,
                                             kernel_size = [16/initialDilation,4/initialDilation],
                                             padding = "same",
                                             activation = tf.nn.relu,
                                             dilation_rate = initialDilation,
                                             strides = 1)
        verticalKernels = tf.layers.conv2d(inputs = imageInput,
                                             filters = 4,
                                             kernel_size = [4/initialDilation,16/initialDilation],
                                             padding = "same",
                                             activation = tf.nn.relu,
                                             dilation_rate = initialDilation,
                                             strides = 1)
        squareKernels = tf.layers.conv2d(inputs = imageInput,
                                             filters = 12,
                                             kernel_size = [8/initialDilation,8/initialDilation],
                                             padding = "same",
                                             activation = tf.nn.relu,
                                             dilation_rate = initialDilation,
                                             strides = 1)
        c1 = tf.concat([horizontalKernels,verticalKernels,squareKernels], axis = 3)
        c1 = tf.layers.max_pooling2d(inputs = c1,
                                     pool_size = 8,
                                     strides = 4,
                                     padding = "same")
        print c1

        numberOfFilters = [10]
        kernelSizes = [8]
        
        poolSizes = [4]
        poolStrides = [4]
        nextInput = c1
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


    def train(self, numberOfExamples, checkpoint = "/tmp/model.checkpoint", restore = False):
        partialImages,targetImages,targetVectors,_ = loadExamples(numberOfExamples, noisy = self.noisy)
        
        initializer = tf.global_variables_initializer()
        iterator = BatchIterator(50,tuple([partialImages,targetImages] + targetVectors),
                                 testingFraction = TESTINGFRACTION, stringProcessor = loadImage)
        iterator.registerPlaceholders([self.currentPlaceholder, self.goalPlaceholder] +
                                      self.decoder.placeholders())
        saver = tf.train.Saver()

        flushEverything()

        with tf.Session() as s:
            if not restore:
                s.run(initializer)
            else:
                saver.restore(s, checkpoint)
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
                flushEverything()

    def analyzeFailures(self, numberOfExamples, checkpoint):
        partialImages,targetImages,targetVectors,targetLines = loadExamples(numberOfExamples,noisy = self.noisy)
        iterator = BatchIterator(1,tuple([partialImages,targetImages] + targetVectors + [targetLines]),
                                 testingFraction = TESTINGFRACTION, stringProcessor = loadImage)
        iterator.registerPlaceholders([self.currentPlaceholder, self.goalPlaceholder] +
                                      self.decoder.placeholders() + [None])
        saver = tf.train.Saver()
        failureLog = [] # pair of current goal
        targetRanks = []
        k = 0

        with tf.Session() as s:
            saver.restore(s,checkpoint)
            for feed in iterator.testingFeeds():
                targetLine = feed[None].reshape((1))[0].tolist()
                del feed[None]
                k += 1
                accuracy = s.run(self.averageAccuracy,
                                 feed_dict = feed)
                assert accuracy == 0.0 or accuracy == 1.0
                if accuracy < 0.5:
                    # decode the action preferred by the model
                    topHundred = self.decoder.beam(s, {self.currentPlaceholder: feed[self.currentPlaceholder],
                                                       self.goalPlaceholder: feed[self.goalPlaceholder]}, 100)
                    topHundred.sort(key = lambda foo: foo[0], reverse = True)
                    preferredLine = topHundred[0][1]
                    preferredLineHumanReadable = str(preferredLine)
                    preferredLine = "\n%end of program\n" if preferredLine == None else preferredLine.TikZ()
                    # check to see the rank of the correct line, because it wasn't the best
                    targetLineString = str(targetLine)
                    topHundred = [ l for _,l in topHundred]
                    topHundredString = map(str,topHundred)
                    
                    if targetLineString in topHundredString:
                        #print "Target line has rank %d in beam"%(1 + topHundred.index(targetLine))
                        targetRanks.append(1 + topHundredString.index(targetLineString))
                    else:
                        print "Target line (not model preference):",targetLine
                        print "Model preference:",preferredLineHumanReadable
                        print "Target not in beam."
                        if isinstance(targetLine, Line):
                            print "The target length = %f"%(targetLine.length())
                            print "Is the target diagonal? %s"%(str(targetLine.isDiagonal()))
                            print "Smallest distance: %f"%(min([targetLine - h for h in topHundred ]))
                        print ""
                        targetRanks.append(None)
                    
                    failureLog.append((feed[self.currentPlaceholder][0], feed[self.goalPlaceholder][0], preferredLine))
                    if len(failureLog) > 99:
                        break
                    
        print "Failures:",len(failureLog),'/',k
        successfulTargetRanks = [ r for r in targetRanks if r != None ]
        print "In beam %d/%d of the time."%(len(successfulTargetRanks),len(targetRanks))
        print "Average successful target rank: %f"%(sum(successfulTargetRanks)/float(len(successfulTargetRanks)))
        print "Successful target ranks: %s"%(str(successfulTargetRanks))
        
        for j,(c,g,l) in enumerate(failureLog):
            saveMatrixAsImage(c*255,"failures/%d-current.png"%j)
            saveMatrixAsImage(g*255,"failures/%d-goal.png"%j)
            pixels = render([l],yieldsPixels = True,canvas = (MAXIMUMCOORDINATE,MAXIMUMCOORDINATE))[0]
            pixels = 1.0 - pixels
            saveMatrixAsImage(pixels*255 + 255*c,"failures/%d-predicted.png"%j)
                

    def SMC(self, targetImage, checkpoint = "/tmp/model.checkpoint", beamSize = 10, beamLength = 10):
        totalNumberOfRenders = 0
        #showImage(targetImage)
        targetImage = np.reshape(targetImage,(256,256))
        beam = [Particle(program = [],
                         output = np.zeros(targetImage.shape),
                         logLikelihood = 0.0,
                         count = beamSize,
                         time = 0.0,
                         distance = asymmetricBlurredDistance(targetImage, np.zeros(targetImage.shape)))]

        finishedPrograms = []
        
        saver = tf.train.Saver()
        with tf.Session() as s:
            saver.restore(s,checkpoint)

            searchStartTime = time()
            
            for iteration in range(beamLength):
                children = []
                startTime = time()
                for parent in beam:
                    

                    childCount = beamSize if self.arguments.beam else parent.count
                    if not self.arguments.unguided:
                        # neural network guide: decoding
                        feed = {self.currentPlaceholder: np.array([parent.output]),
                                self.goalPlaceholder: np.array([targetImage])}
                        kids = self.decoder.beam(s, feed, childCount*2)
                    else:
                        # no neural network guide: sample from the prior
                        kids = [ (0.0, randomLineOfCode()) for _ in range(childCount) ]
                        
                    kids.sort(key = lambda k: k[0], reverse = True)
                    for childScore,suffix in kids[:childCount]:
                        if suffix == None:
                            k = Sequence(parent.program)
                        else:
                            k = parent.program + [suffix]
                        children.append(Particle(program = k,
                                                 logLikelihood = parent.logLikelihood + childScore,
                                                 count = 1,
                                                 parent = parent,
                                                 time = time() - searchStartTime))
                
                print "Ran neural network beam in %f seconds"%(time() - startTime)

                beam = children
                
#                beam = self.removeParticlesWithCollisions(beam)
                if self.arguments.beam:
                    beam = sorted(beam, key = lambda p: p.logLikelihood,reverse = True)[:beamSize]
                assert len(beam) <= beamSize
                self.renderParticles(beam)
                totalNumberOfRenders += len(beam)

                print "Iteration %d: %d total renders.\n"%(iteration+1,totalNumberOfRenders)

                for n in beam:
                    n.distance = asymmetricBlurredDistance(targetImage, n.output)

                # record/remove all of the finished programs
                finishedPrograms += [ n for n in beam if n.finished() ]
                beam = [ n for n in beam if not n.finished() ]                

                # Resample
                for n in beam:
                    n.score = 0.0
                    if n.parent.count > 0:
                        n.score += math.log(n.parent.count) # simulate affect of drawing repeatedly from previous distribution
                    else:
                        assert self.arguments.beam # should only occur in straight up beam search
                    n.score += self.arguments.proposalCoefficient *(n.logLikelihood)
                    n.score += self.arguments.distanceCoefficient *(- n.distance)
                    n.score += self.arguments.parentCoefficient   *(n.parent.distance)
                    n.score += self.arguments.priorCoefficient    *(n.program[-1].logPrior())
                    n.score /= self.arguments.temperature
                    
                z = lseList([ n.score for n in beam ])
                ps = np.array([math.exp(n.score - z) for n in beam ])
                cs = np.random.multinomial(beamSize, ps/ps.sum()).tolist()
                for n,c in zip(beam,cs):
                    n.count = c

                beam = self.consolidateIdenticalParticles(beam)

                for n in beam:
                    if n.count == 0 and not self.arguments.beam: continue
                    
                    p = n.program
                    if not n.finished(): p = Sequence(p)
                    print "(x%d) Program in beam (%f):\n%s"%(n.count, n.logLikelihood, str(p))
                    print "Blurred distance: %f"%n.distance
                    if n.count > beamSize/5 and iteration > 4 and False:
                        showImage(n.output + targetImage)
                    print "\n"
                
                # Remove all of the dead particles, and less were doing a straight beam decoding
                if not self.arguments.beam:
                    beam = [ n for n in beam if n.count > 0 ]
                if beam == []:
                    print "Empty beam."
                    break
            return finishedPrograms

    # helper functions for particle search
    def removeParticlesWithCollisions(self,particles):
        return [ n for n in particles
                 if not (n.program if n.finished() else Sequence(n.program)).hasCollisions() ]
    def consolidateIdenticalParticles(self,particles):
        consolidated = []
        for p in sorted(particles,key = lambda p: -p.logLikelihood):
            duplicate = False
            for c in consolidated:
                if np.array_equal(p.output, c.output):
                    c.count += p.count
                    c.logLikelihood = max([c.logLikelihood,p.logLikelihood])
                    duplicate = True
                    break
            if not duplicate: consolidated.append(p)
        return consolidated
    
    def renderParticles(self,particles):
        startTime = time()
        if not self.arguments.fastRender:
            outputs = render([ (n.program if n.finished() else Sequence(n.program)).TikZ()
                               for n in particles ],
                             yieldsPixels = True,
                             canvas = (MAXIMUMCOORDINATE,MAXIMUMCOORDINATE))
        else:
            outputs = [ fastRender(n.program if n.finished() else Sequence(n.program))
                        for n in particles ]
            
        print "Rendered in %f seconds"%(time() - startTime)
        for n,o in zip(particles,outputs):
            n.output = o
            if not self.arguments.fastRender:
                n.output = 1.0 - n.output
    def saveParticles(self,finishedPrograms, parseDirectory, targetImage):
        print "Finished programs, sorted by likelihood:"
        os.system('rm -r %s'%(parseDirectory))
        os.system('mkdir %s'%(parseDirectory))
        finishedPrograms.sort(key = lambda n: -n.logLikelihood)
        for j,n in enumerate(finishedPrograms):
            n.parent = None
            print "Finished program: log likelihood %f"%(n.logLikelihood)
            print n.program
            saveMatrixAsImage(n.output*255, "%s/%d.png"%(parseDirectory, j))
            pickle.dump(n, open("%s/particle%d.p"%(parseDirectory, j),'w'))
            print "Distance: %f"%(n.distance)
            #asymmetricBlurredDistance(targetImage, n['output'], True)
            print ""

        
    def visualizeFilters(self,checkpoint):
        filters = []
        saver = tf.train.Saver()
        with tf.Session() as s:
            saver.restore(s,checkpoint)
            print tf.GraphKeys.TRAINABLE_VARIABLES
            for v in tf.trainable_variables():
                if v.name.startswith("conv2d") and v.name.endswith('kernel:0'):
                    print v
                    filters.append(v.eval())
        # first layer
        filters = filters[:3]
        for f in filters:
            (w,h,_,n) = f.shape
            print f.shape

            imageWidth = (w)*n + 5*(n - 1)
            imageHeight = (2*h + 5)
            v = np.zeros((imageWidth,imageHeight))
            print "v = ",v.shape
            for j in range(n):
                print "target size",v[w*j:w*(j+1), 0:h].shape
                print "destination size",f[:,:,0,j].shape
                v[(w+5)*j:(w+5)*j+w, 0:h] = f[:,:,0,j]
                v[(w+5)*j:(w+5)*j+w, h+5:2*h+5] = f[:,:,1,j]
            saveMatrixAsImage(v*255,"/tmp/filters.png")
            os.system("feh /tmp/filters.png")

def handleTest(a):
    (f,arguments) = a
    tf.reset_default_graph()
    model = RecognitionModel(arguments)
    targetImage = loadImage(f)
    particles = model.SMC(targetImage,
                          beamSize = arguments.beamWidth,
                          beamLength = arguments.beamLength,
                          checkpoint = arguments.checkpoint)
    # place where we will save the parses
    parseDirectory = targetImage[:-4] + "-parses"
    model.saveParticles(particles, parsedDirectory, targetImage)
    
def picturesInDirectory(d):
    if d.endswith('.png'): return [d]
    if not d.endswith('/'): d = d + '/'
    return [ d + f for f in os.listdir(d) if f.endswith('.png') ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'training and evaluation of recognition models')
    parser.add_argument('task')
    parser.add_argument('-c','--checkpoint', default = "checkpoints/model.checkpoint", type = str)
    parser.add_argument('-n','--numberOfExamples', default = 100000, type = int)
    parser.add_argument('-l','--beamLength', default = 13, type = int)
    parser.add_argument('-b','--beamWidth', default = 10, type = int)
    parser.add_argument('-t','--test', default = '', type = str)
    parser.add_argument('-r', action="store_true", default=False)
    parser.add_argument('-m','--cores', default = 1, type = int)
    parser.add_argument('--noisy',action = "store_true", default = False)

    # parameters of sequential Monte Carlo
    parser.add_argument('-T','--temperature', default = 1.0, type = float)
    parser.add_argument('--parentCoefficient', default = 0.0, type = float)
    parser.add_argument('--proposalCoefficient', default = 0.0, type = float)
    parser.add_argument('--distanceCoefficient', default = 1.0/25.0, type = float)
    parser.add_argument('--priorCoefficient', default = 0.0, type = float)
    parser.add_argument('--beam', action = "store_true", default = False)
    parser.add_argument('--fastRender', action = "store_true", default = False)
    parser.add_argument('--unguided', action = "store_true", default = False)
    

    arguments = parser.parse_args()
    if arguments.fastRender:
        loadPrecomputedRenderings()
    
    if arguments.task == 'test':
        fs = picturesInDirectory(arguments.test)
        if arguments.cores == 1:
            map(handleTest, [ (f,arguments) for f in fs ])
        else:
            Pool(arguments.cores).map(handleTest, [ (f,arguments) for f in fs ])
    
    elif arguments.task == 'visualize':
        RecognitionModel(arguments).visualizeFilters(arguments.checkpoint)
    elif arguments.task == 'analyze':
        RecognitionModel(arguments).analyzeFailures(arguments.numberOfExamples, checkpoint = arguments.checkpoint)
    elif arguments.task == 'train':
        RecognitionModel(arguments).train(arguments.numberOfExamples, checkpoint = arguments.checkpoint, restore = arguments.r)
    elif arguments.task == 'profile':
        cProfile.run('loadExamples(%d)'%(arguments.numberOfExamples))
