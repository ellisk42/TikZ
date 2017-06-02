from distanceExamples import *
from architectures import architectures
from batch import BatchIterator
from language import *
from render import render,animateMatrices
from utilities import *
from distanceMetrics import blurredDistance,asymmetricBlurredDistance,analyzeAsymmetric
from makeSyntheticData import randomScene
from groundTruthParses import getGroundTruthParse
from loadTrainingExamples import *

import argparse
import sys
import tensorflow as tf
import os
import io
from time import time
import pickle
import cProfile
from multiprocessing import Pool
import random

# The data is generated on a  MAXIMUMCOORDINATExMAXIMUMCOORDINATE grid
# We can interpolate between stochastic search and neural networks by downsampling to a smaller grid
APPROXIMATINGGRID = MAXIMUMCOORDINATE
def coordinate2grid(c): return c*MAXIMUMCOORDINATE/APPROXIMATINGGRID
def grid2coordinate(g): return g*APPROXIMATINGGRID/MAXIMUMCOORDINATE

TESTINGFRACTION = 0.05

[STOP,CIRCLE,LINE,RECTANGLE,LABEL] = range(5)


class StandardPrimitiveDecoder():
    def makeNetwork(self,imageRepresentation):
        # A placeholder for each target
        self.targetPlaceholder = [ (tf.placeholder(tf.int32, [None]) if d >= 0
                                    else tf.placeholder(tf.float32, [None]))
                                   for d in self.outputDimensions ]
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
#            if :
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
        self.outputDimensions = [APPROXIMATINGGRID,APPROXIMATINGGRID,APPROXIMATINGGRID] # x,y,r
        self.hiddenSizes = [None, None, None]
        self.makeNetwork(imageRepresentation)
    
    def beam(self, session, feed, beamSize):
        return [(s, Circle(AbsolutePoint(grid2coordinate(x),grid2coordinate(y)),r))
                for s,[x,y,r] in self.beamTrace(session, feed, beamSize)
                if x > 1 and y > 1 and x < MAXIMUMCOORDINATE - 1 and y < MAXIMUMCOORDINATE - 1]

    @staticmethod
    def extractTargets(l):
        if l != None and isinstance(l,Circle):
            return [coordinate2grid(l.center.x),
                    coordinate2grid(l.center.y),
                    coordinate2grid(l.radius)]
        return [0,0,0]

class LabelDecoder(StandardPrimitiveDecoder):
    token = LABEL
    languagePrimitive = Label
    
    def __init__(self, imageRepresentation):
        self.outputDimensions = [APPROXIMATINGGRID,APPROXIMATINGGRID,len(Label.allowedLabels)] # x,y,c
        self.hiddenSizes = [None, None, None]
        self.makeNetwork(imageRepresentation)
    
    def beam(self, session, feed, beamSize):
        return [(s, Label(AbsolutePoint(grid2coordinate(x),grid2coordinate(y)),Label.allowedLabels[l]))
                for s,[x,y,l] in self.beamTrace(session, feed, beamSize)
                if x > 1 and y > 1 and x < MAXIMUMCOORDINATE - 1 and y < MAXIMUMCOORDINATE - 1]

    @staticmethod
    def extractTargets(l):
        if l != None and isinstance(l,Label):
            return [coordinate2grid(l.p.x),
                    coordinate2grid(l.p.y),
                    coordinate2grid(Label.allowedLabels.index(l.c))]
        return [0,0,0]

class RectangleDecoder(StandardPrimitiveDecoder):
    token = RECTANGLE
    languagePrimitive = Rectangle

    def __init__(self, imageRepresentation):
        self.outputDimensions = [APPROXIMATINGGRID,APPROXIMATINGGRID,APPROXIMATINGGRID,APPROXIMATINGGRID] # x,y
        self.hiddenSizes = [None,
                            None,
                            None,
                            None]
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
        if l != None and isinstance(l,Rectangle):
            return [coordinate2grid(l.p1.x),
                    coordinate2grid(l.p1.y),
                    coordinate2grid(l.p2.x),
                    coordinate2grid(l.p2.y)]
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
        return [(s, Line.absolute((grid2coordinate(x1)),
                                  (grid2coordinate(y1)),
                                  (grid2coordinate(x2)),
                                  (grid2coordinate(y2)),
                                  arrow = arrow == 1,solid = solid == 1))
                for s,[x1,y1,x2,y2,arrow,solid] in self.beamTrace(session, feed, beamSize)
                if (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) > 0 and x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0 ]

    @staticmethod
    def extractTargets(l):
        if l != None and isinstance(l,Line):
            return [coordinate2grid(l.points[0].x),
                    coordinate2grid(l.points[0].y),
                    coordinate2grid(l.points[1].x),
                    coordinate2grid(l.points[1].y),
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
    # It shouldn't matter in what order these are listed. If it does then I will consider that a bug.
    decoderClasses = [CircleDecoder,
                      RectangleDecoder,
                      LineDecoder,
                      LabelDecoder,
                      StopDecoder]
    def __init__(self, imageRepresentation, trainingPredicatePlaceholder):
        self.decoders = [k(imageRepresentation) for k in PrimitiveDecoder.decoderClasses]

        self.prediction = tf.layers.dense(imageRepresentation, len(self.decoders))
        self.hard = tf.cast(tf.argmax(self.prediction,dimension = 1),tf.int32)
        self.soft = tf.nn.log_softmax(self.prediction)
        self.targetPlaceholder = tf.placeholder(tf.int32, [None])
        self.imageRepresentation = imageRepresentation
        self.trainingPredicatePlaceholder = trainingPredicatePlaceholder

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
            if l != None and isinstance(l,d.languagePrimitive):
                t = [d.token]
                break
        for d in PrimitiveDecoder.decoderClasses:
            t += d.extractTargets(l)
        return t

    def beam(self, session, feed, beamSize):
        feed[self.trainingPredicatePlaceholder] = False
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

class RecurrentDecoder():
    def __init__(self, imageFeatures):
        self.unit = LSTM(imageFeatures)

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
    # wraps it up in a sequence object if it hasn't already
    def sequence(self):
        if self.finished(): return self.program
        return Sequence(self.program)
    def render(self):
        if self.output is None:
            self.output = self.sequence().draw()
        return self.output

class RecognitionModel():
    def __init__(self, arguments):
        self.noisy = arguments.noisy
        self.arguments = arguments
        self.graph = tf.Graph()
        self.session = tf.Session(graph = self.graph)
        with self.session.graph.as_default():
            # current and goal images
            self.currentPlaceholder = tf.placeholder(tf.float32, [None, 256, 256])
            self.goalPlaceholder = tf.placeholder(tf.float32, [None, 256, 256])

            self.trainingPredicatePlaceholder = tf.placeholder(tf.bool)

            imageInput = tf.stack([self.currentPlaceholder,self.goalPlaceholder], axis = 3)

            c1 = architectures[self.arguments.architecture].makeModel(imageInput)
            c1d = int(c1.shape[1]*c1.shape[2]*c1.shape[3])
            print "fully connected input dimensionality:",c1d

            f1 = tf.reshape(c1, [-1, c1d])
            if self.arguments.dropout:
                f1 = tf.layers.dropout(f1,
                                       training = self.trainingPredicatePlaceholder)

            self.decoder = PrimitiveDecoder(f1, self.trainingPredicatePlaceholder)
            self.loss = self.decoder.loss()
            self.averageAccuracy = self.decoder.accuracy()

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.arguments.learningRate).minimize(self.loss)

    def loadCheckpoint(self, checkpoint):
        with self.session.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.session, checkpoint)
            
    def train(self, numberOfExamples, checkpoint = "/tmp/model.checkpoint", restore = False):
        noisyTarget,programs = loadExamples(numberOfExamples)
        
        iterator = BatchIterator(10,(np.array(noisyTarget),np.array(programs)),
                                 testingFraction = TESTINGFRACTION, stringProcessor = loadImage)
        flushEverything()

        with self.session.graph.as_default():
            initializer = tf.global_variables_initializer()
            saver = tf.train.Saver()

            if not restore:
                self.session.run(initializer)
            else:
                saver.restore(self.session, checkpoint)
            
            for e in range(20):
                epicLoss = []
                epicAccuracy = []
                for ts,ps in iterator.epochExamples():
                    feed = self.makeTrainingFeed(ts,ps)
                    feed[self.trainingPredicatePlaceholder] = True
                    _,l,accuracy = self.session.run([self.optimizer, self.loss, self.averageAccuracy],
                                         feed_dict = feed)
                    epicLoss.append(l)
                    epicAccuracy.append(accuracy)
                print "Epoch %d: accuracy = %f, loss = %f"%((e+1),sum(epicAccuracy)/len(epicAccuracy),sum(epicLoss)/len(epicLoss))
                testingAccuracy = []
                for ts,ps in iterator.testingExamples():
                    feed = self.makeTrainingFeed(ts,ps)
                    feed[self.trainingPredicatePlaceholder] = False
                    testingAccuracy.append(self.session.run(self.averageAccuracy, feed_dict = feed))
                print "\tTesting accuracy = %f"%(sum(testingAccuracy)/len(testingAccuracy))
                print "Saving checkpoint: %s" % saver.save(self.session, checkpoint)
                flushEverything()

    def makeTrainingFeed(self, targets, programs):
        # goal, current, predictions
        gs = []
        cs = []
        ps = []
        for target, program in zip(targets, programs):
            if not self.arguments.noisy:
                target = program.draw()
            for j in range(len(program) + 1):
                gs.append(target)
                cs.append(Sequence(program.lines[:j]).draw())
                l = None
                if j < len(program): l = program.lines[j]
                ps.append(self.decoder.extractTargets(l))

        gs = np.array(gs)
        cs = np.array(cs)
        ps = np.array(ps)
        
        if self.arguments.noisy: gs = augmentData(gs)
        
        f = {self.goalPlaceholder: gs,
             self.currentPlaceholder: cs}
        for j,p in enumerate(self.decoder.placeholders()):
            f[p] = ps[:,j]
        return f

    def beam(self, current, goal, beamSize):
        feed = {self.currentPlaceholder: np.array([current]),
                self.goalPlaceholder: np.array([goal])}
        return self.decoder.beam(self.session, feed, beamSize)

    
                    
class DistanceModel():
    def __init__(self,arguments):
        self.arguments = arguments
        self.graph = tf.Graph()
        self.session = tf.Session(graph = self.graph)
        with self.session.graph.as_default():
            # current and goal images
            self.currentPlaceholder = tf.placeholder(tf.float32, [None, 256, 256])
            self.goalPlaceholder = tf.placeholder(tf.float32, [None, 256, 256])

            imageInput = tf.stack([self.currentPlaceholder,self.goalPlaceholder], axis = 3)

            c1 = architectures[self.arguments.architecture].makeModel(imageInput)
            c1d = int(c1.shape[1]*c1.shape[2]*c1.shape[3])
            print "fully connected input dimensionality:",c1d

            f1 = tf.reshape(c1, [-1, c1d])

            # Value function learning
            self.valueTargets = tf.placeholder(tf.float32, [None,2]) # (extra target, extra current)
            # this line of code collapses all of the filters into batchSize*numberOfFilters
            #f2 = tf.reduce_sum(c1, [1,2])
            f2 = f1
            self.distanceFunction = tf.layers.dense(f2, 2, activation = tf.nn.relu)
            self.distanceLoss = tf.reduce_mean(tf.squared_difference(self.valueTargets, self.distanceFunction))
            self.distanceOptimizer = tf.train.AdamOptimizer(learning_rate=self.arguments.learningRate).minimize(self.distanceLoss)

    def learnedDistances(self, currentBatch, goalBatch):
        return self.session.run(self.distanceFunction,
                                feed_dict = {self.currentPlaceholder: currentBatch,
                                             self.goalPlaceholder: goalBatch})
    
    def learnedParticleDistances(self, goal, particles):
        if particles == []: return 
        # only do it for 50 particles at a time
        maximumBatchSize = 50
        if len(particles) > maximumBatchSize:
            self.learnedParticleDistances(goal, particles[maximumBatchSize:])
        particles = particles[:maximumBatchSize]
        d = self.learnedDistances(np.array([ p.render() for p in particles ]),
                                  np.tile(goal, (len(particles), 1, 1)))
        for j,p in enumerate(particles):
            if self.arguments.showParticles:
                print "Distance vector:",d[j,:]
                print "Likelihood:",p.logLikelihood
                showImage(p.output + goal)
            p.distance = (d[j,0], d[j,1])

    def closedSession(self): self.session.close()

    def loadCheckpoint(self, checkpoint):
        with self.session.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.session, checkpoint)
        
    def train(self, numberOfExamples, checkpoint, restore = False):
        assert self.arguments.noisy
        targetImages,targetPrograms = loadExamples(numberOfExamples)
        iterator = BatchIterator(5,tuple([np.array(targetImages),np.array(targetPrograms)]),
                                testingFraction = TESTINGFRACTION, stringProcessor = loadImage)

        # use the session to make sure that we save or initialize the right things
        with self.session.graph.as_default():
            initializer = tf.global_variables_initializer()
            saver = tf.train.Saver()
            flushEverything()
            if not restore:
                self.session.run(initializer)
            else:
                saver.restore(self.session, checkpoint)
            for e in range(20):
                runningAverage = 0.0
                runningAverageCount = 0
                lastUpdateTime = time()
                for images,programs in iterator.epochExamples():
                    targets, current, distances = makeDistanceExamples(images, programs,
                                                                       reportTime = runningAverageCount == 0)
                    _,l = self.session.run([self.distanceOptimizer, self.distanceLoss],
                                feed_dict = {self.currentPlaceholder: current,
                                             self.goalPlaceholder: targets,
                                             self.valueTargets: distances})
                    runningAverage += l
                    runningAverageCount += 1
                    if time() - lastUpdateTime > 120:
                        lastUpdateTime = time()
                        print "\t\tRunning average loss: %f"%(runningAverage/runningAverageCount)
                        flushEverything()
                print "Epoch %d: loss = %f"%(e,runningAverage/runningAverageCount)
                flushEverything()

                testingLosses = [ self.session.run(self.distanceLoss,
                                        feed_dict = {self.currentPlaceholder: current,
                                                     self.goalPlaceholder: targets,
                                                     self.valueTargets: distances})
                                  for images,programs in iterator.testingExamples()
                                  for [targets, current, distances] in [makeDistanceExamples(images, programs)] ]
                testingLosses = sum(testingLosses)/len(testingLosses)
                print "\tTesting loss: %f"%testingLosses
                print "Saving checkpoint: %s"%(saver.save(self.session, checkpoint))
                flushEverything()

    def analyzeGroundTruth(self):
        self.loadCheckpoint(self.arguments.distanceCheckpoint)
        targetNames = [ "drawings/expert-%d.png"%j for j in range(100) ]
        targetImages = map(loadImage,targetNames)
        targetSequences = map(getGroundTruthParse,targetNames)

        for j in range(100):
            d = self.learnedDistances(np.array([targetSequences[j].draw()]),
                                      np.array([targetImages[j]]))[0]
            print "%f\t%f"%(d[0],d[1])
            if d[0] > 0.5 or d[1] > 0.5:
                showImage(targetImages[j])
        

class SearchModel():
    def __init__(self,arguments):
        self.arguments = arguments
        self.recognizer = RecognitionModel(arguments)
        self.distance = DistanceModel(arguments)

        # load the networks
        self.recognizer.loadCheckpoint(self.arguments.checkpoint)
        self.distance.loadCheckpoint(self.arguments.distanceCheckpoint)

    def SMC(self, targetImage, beamSize = 10, beamLength = 10):
        totalNumberOfRenders = 0
        targetImage = np.reshape(targetImage,(256,256))
        beam = [Particle(program = [],
                         output = np.zeros(targetImage.shape),
                         logLikelihood = 0.0,
                         count = beamSize,
                         time = 0.0,
                         distance = 999999999)]

        finishedPrograms = []
        
        searchStartTime = time()

        for iteration in range(beamLength):
            lastIteration = iteration == beamLength - 1 # are we the last iteration
            children = []
            startTime = time()
            for parent in beam:
                childCount = beamSize if self.arguments.beam else parent.count
                if not self.arguments.unguided:
                    # neural network guide: decoding
                    kids = self.recognizer.beam(parent.output, targetImage, childCount)
                else:
                    # no neural network guide: sample from the prior
                    kids = [ (0.0, randomCode) for _ in range(childCount)
                             for randomCode in [randomLineOfCode()] if randomCode != None ]
                    kids += [(0.0, None)]

                # remove children that duplicate an existing line of code
                existingLinesOfCode = map(str,parent.program)
                kids = [ child for child in kids
                         if not (str(child[1]) in existingLinesOfCode) ] 
                kids.sort(key = lambda k: k[0], reverse = True)
                
                # in evaluation mode we want to make sure that there are at least some finished programs
                if self.arguments.task == 'evaluate' and (not [ k for k in kids[:childCount] if k[1] == None]) and childCount > 1:
                    kids = [ k for k in kids if k[1] == None] + kids
                    
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

            if lastIteration and self.arguments.task != 'evaluate': children = [p for p in children if p.finished() ]
                
            if not self.arguments.quiet:
                print "Ran neural network beam in %f seconds"%(time() - startTime)

            beam = children

            if self.arguments.noIntersections:
                beam = self.removeParticlesWithCollisions(beam)
            if self.arguments.beam:
                beam = sorted(beam, key = lambda p: p.logLikelihood,reverse = True)[:beamSize]
            assert len(beam) <= beamSize
            self.renderParticles(beam)
            totalNumberOfRenders += len(beam)

            if not self.arguments.quiet:
                print "Iteration %d: %d total renders.\n"%(iteration+1,totalNumberOfRenders)

            if self.arguments.distance: # use the learned distance metric
                self.distance.learnedParticleDistances(targetImage, beam)
                for p in beam: p.distance = p.distance[0] + self.arguments.mistakePenalty*p.distance[1]
            else:
                for n in beam:
                    if self.arguments.task == 'evaluate':
                        # evaluation is  on noiseless data
                        assert not self.arguments.noisy
                        difference = targetImage - n.output
                        n.distance = np.sum(np.abs(difference[difference > 0])) + 20*np.sum(np.abs(difference[difference < 0]))
                    else:
                        n.distance = asymmetricBlurredDistance(targetImage, n.output)

            if not self.arguments.quiet: print "Computed distances"

            # record/remove all of the finished programs
            finishedPrograms += [ n for n in beam if n.finished() ]
            beam = [ n for n in beam if not n.finished() ]

            if beam == []: break            

            # Resample
            for n in beam:
                n.score = 0.0
                if n.parent.count > 0:
                    n.score += math.log(n.parent.count) # simulate affect of drawing repeatedly from previous distribution
                else:
                    assert self.arguments.beam # should only occur in straight up beam search
                n.score += self.arguments.proposalCoefficient *(n.logLikelihood - n.parent.logLikelihood)
                n.score += self.arguments.distanceCoefficient *(- n.distance)
                if self.arguments.parentCoefficient:
                    n.score += self.arguments.distanceCoefficient   *(n.parent.distance)
                n.score += self.arguments.priorCoefficient    *(n.program[-1].logPrior())
                n.score /= self.arguments.temperature

            z = lseList([ n.score for n in beam ])
            ps = np.array([math.exp(n.score - z) for n in beam ])
            cs = np.random.multinomial(beamSize, ps/ps.sum()).tolist()
            for n,c in zip(beam,cs):
                n.count = c

            if not self.arguments.quiet: print "Resampled."

            # Remove all of the dead particles, and less were doing a straight beam decoding
            if not self.arguments.beam:
                beam = [ n for n in beam if n.count > 0 ]

            beam = self.consolidateIdenticalParticles(beam)

            if not self.arguments.quiet:
                for n in beam:
                    p = n.program
                    if not n.finished(): p = Sequence(p)
                    print "(x%d) Program in beam (%f):\n%s"%(n.count, n.logLikelihood, str(p))
                    print "Blurred distance: %f"%n.distance
                    if n.count > beamSize/5 and self.arguments.showParticles:
                        showImage(n.output + targetImage)
                    print "\n"

            if beam == []:
                print "Empty beam."
                break


        if finishedPrograms == []:
            print "No finished programs!"
            # showImage(targetImage)
            # for p in beam:
            #     showImage(p.output)
        # Remove these pointers so that they can be garbage collected. Replaced distance with something more meaningful.
        for p in finishedPrograms:
#            p.distance = None
            p.output = None
            p.parent = None
        return finishedPrograms

    # helper functions for particle search
    def removeParticlesWithCollisions(self,particles):
        return [ n for n in particles
                 if not (n.program if n.finished() else Sequence(n.program)).hasCollisions() ]
    def consolidateIdenticalParticles(self,particles):
        # we need to do this in linear time
        startTime = time()
        startingParticles = len(particles)
        # map from the hash code of an image to the particles with that hash
        particleMap = {}
        for p in particles:
            p.output.flags.writeable = False
            p.outputHash = hash(p.output.data)
            if not (p.outputHash in particleMap):
                particleMap[p.outputHash] = [p]
                continue

            collision = None # None = no collision, True = collided better, particle = remove it
            for q in particleMap[p.outputHash]:
                if np.array_equal(p.output, q.output) and p.finished() == q.finished():
                    # collision
                    if p.logLikelihood > q.logLikelihood: # prefer p
                        p.count += q.count
                        collision = q
                    else:
                        collision = True
                        q.count += p.count
                    break
            if collision  == True: continue
            particleMap[p.outputHash].append(p)
            if collision != None:
                particleMap[p.outputHash].remove(collision)

        finalParticles = [ p for ps in particleMap.values() for p in ps  ]
        if not self.arguments.quiet:
            print "Consolidated %d particles into %d particles in %f seconds."%(startingParticles,
                                                                                len(finalParticles),
                                                                                time() - startTime)
        return finalParticles
    
    def renderParticles(self,particles):
        startTime = time()
        assert self.arguments.fastRender
        # optimization: add the rendering of last command to the parent
        # todo: get that optimization working for Cairo if it does being important
        for n in particles: n.output = n.sequence().draw()
        if not self.arguments.quiet: print "Rendered in %f seconds"%(time() - startTime)

    def saveParticles(self,finishedPrograms, parseDirectory, targetImage):
        print "Finished programs, sorted by likelihood:"
        if os.path.exists(parseDirectory):
            os.system('rm -rf %s/*'%(parseDirectory))
        else:
            os.system('mkdir %s'%(parseDirectory))
        self.learnedParticleDistances(targetImage, finishedPrograms)
        likelihoodCoefficient = 0.3
        distanceCoefficient = (-7,-7)
        priorCoefficient = 0.05
        finishedPrograms.sort(key = lambda n: likelihoodCoefficient*n.logLikelihood + distanceCoefficient[0]*n.distance[0] + distanceCoefficient[1]*n.distance[1] + priorCoefficient*n.program.logPrior(),
                              reverse = True)
        for j,n in enumerate(finishedPrograms[:500]):
            n.parent = None
            n.output = None
            if j < 10:
                print "Finished program: log likelihood %f"%(n.logLikelihood)
                print n.program
                print "Distance: %s"%(str(n.distance))
                print ""

            saveMatrixAsImage(fastRender(n.program)*255, "%s/%d.png"%(parseDirectory, j))
            pickle.dump(n, open("%s/particle%d.p"%(parseDirectory, j),'wb'))


    def evaluateAccuracy(self):
        # similar map but for the rank of the correct program
        programRank = {}
        # distance from correct program to suggested program
        programDistance = {}
        # intersection of reunion
        intersectionDistance = {}
        # how long the search took
        searchTime = {}

        # generate target programs with the same random seed so that
        # we get consistent validation sets across models
        random.seed(42)
        targetPrograms = [ randomScene(16)() for _ in range(self.arguments.numberOfExamples) ]
        
        for targetProgram in targetPrograms:
            targetImage = targetProgram.draw()
            k = len(targetProgram.lines)
            if not k in intersectionDistance:
                intersectionDistance[k] = []
                programRank[k] = []
                programDistance[k] = []
                searchTime[k] = []
            
            startTime = time()
            particles = self.SMC(targetImage,
                                 beamSize = arguments.beamWidth,
                                 beamLength = k + 1)
            searchTime[k].append(time() - startTime)
            if particles == []:
                print "No solutions."
                intersectionDistance[k].append(None)
                programRank[k].append(None)
                programDistance[k].append(None)
                continue

            # Sort the particles. Our preference depends on how the search was done.
            if self.arguments.beam:
                preference = lambda p: p.logLikelihood
            elif self.arguments.unguided:
                preference = lambda p: p.program.logPrior() - p.distance
            else: # guided Monte Carlo
                preference = lambda p: p.logLikelihood - p.distance*0.1
            particles.sort(key = preference,reverse = True)
            
            # find the intersection distance of the preferred particle
            preferred = particles[0]
            targetSet = set(map(str,targetProgram.lines))
            preferredSet = set(map(str,preferred.program.lines))
            intersectionDistance[k].append(len(targetSet&preferredSet)/float(len(targetSet|preferredSet)))

            # find the program distance of the preferred particle
            programDistance[k].append(len(targetSet^preferredSet))
            
            # see if any of the programs match exactly
            
            rank = None
            for r,p in enumerate(particles):
                if set(map(str,p.program.lines)) == targetSet:
                    rank = r + 1
                    print "Rank: %d"%(r + 1)
                    break
            programRank[k].append(rank)

        print programRank
        print intersectionDistance
        print programDistance
        print searchTime

        n = len([ r for rs in programRank.values() for r in rs ])
        ranks = [ r for rs in programRank.values() for r in rs if r != None]
        programDistances = [ r for rs in programDistance.values() for r in rs if r != None]
        intersectionDistances = [ r for rs in intersectionDistance.values() for r in rs if r != None]
        print "Got the correct program %d/%d times"%(len(ranks),n)
        print "Average program distance: %f"%(sum(programDistances)/float(len(programDistances)))
        print "Average intersection distance: %f"%(sum(intersectionDistances)/float(len(intersectionDistances)))
        

def handleTest(a):
    (f,arguments) = a
    tf.reset_default_graph()
    model = SearchModel(arguments)
    targetImage = loadImage(f)

    # l = 0 implies that we should look at the ground truth and use that to abound the length
    l = arguments.beamLength
    if l == 0:
        l = len(getGroundTruthParse(f).lines) + 1        
    particles = model.SMC(targetImage,
                          beamSize = arguments.beamWidth,
                          beamLength = l)
    gotGroundTruth = None
    groundTruth = getGroundTruthParse(f)
    if groundTruth != None:
        groundTruth = set(map(str,groundTruth.lines))
        gotGroundTruth = False
        for p in particles:
            if len(set(map(str,p.sequence().lines)) ^ groundTruth) == 0:
                print "Got ground truth for %s"%f
                gotGroundTruth = True
                break
        if not gotGroundTruth:
            print "Did not get ground truth for %s"%f
    # place where we will save the parses
    parseDirectory = f[:-4] + "-parses"
    model.saveParticles(particles, parseDirectory, targetImage)
    return gotGroundTruth
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'training and evaluation of recognition models')
    parser.add_argument('task')
    parser.add_argument('-c','--checkpoint', default = "checkpoints/model.checkpoint", type = str)
    parser.add_argument('-d','--distanceCheckpoint', default = "checkpoints/distance.checkpoint", type = str)
    parser.add_argument('-n','--numberOfExamples', default = 100000, type = int)
    parser.add_argument('-l','--beamLength', default = 13, type = int)
    parser.add_argument('-b','--beamWidth', default = 10, type = int)
    parser.add_argument('-t','--test', default = '', type = str)
    parser.add_argument('-r', action="store_true", default=False)
    parser.add_argument('-m','--cores', default = 1, type = int)
    parser.add_argument('--noisy',action = "store_true", default = False)
    parser.add_argument('--quiet',action = "store_true", default = False)
    parser.add_argument('--dropout',action = "store_true", default = False)
    parser.add_argument('--distance',action = "store_true", default = False)
    parser.add_argument('--learningRate', default = 0.001, type = float)
    parser.add_argument('--architecture', default = "original", type = str)    

    # parameters of sequential Monte Carlo
    parser.add_argument('-T','--temperature', default = 1.0, type = float)
    parser.add_argument('--proposalCoefficient', default = 1.0, type = float)
    parser.add_argument('--parentCoefficient', action = "store_true", default = False)
    parser.add_argument('--distanceCoefficient', default = 1.0/25.0, type = float)
    parser.add_argument('--priorCoefficient', default = 0.0, type = float)
    parser.add_argument('--mistakePenalty', default = 5.0, type = float)
    parser.add_argument('--beam', action = "store_true", default = False)
    parser.add_argument('--fastRender', action = "store_true", default = True)
    parser.add_argument('--unguided', action = "store_true", default = False)
    parser.add_argument('--noIntersections', action = "store_true", default = False)
    parser.add_argument('--showParticles', action = "store_true", default = False)

    arguments = parser.parse_args()

    if arguments.task == 'evaluate':
        assert 'clean' in arguments.checkpoint
    
    if arguments.task == 'showSynthetic':
        print "not implemented"
    elif arguments.task == 'test':
        fs = picturesInDirectory(arguments.test)
        if arguments.cores == 1:
            gt = map(handleTest, [ (f,arguments) for f in fs ])
        else:
            gt = Pool(arguments.cores).map(handleTest, [ (f,arguments) for f in fs ])
        gt = [ g for g in gt if g != None ]
        print "Got a ground truth parse correct %f"%(float(len([None for g in gt if g ]))/float(len(gt)))
    
    elif arguments.task == 'visualize':
        RecognitionModel(arguments).visualizeFilters(arguments.checkpoint)
    elif arguments.task == 'analyze':
        RecognitionModel(arguments).analyzeFailures(arguments.numberOfExamples, checkpoint = arguments.checkpoint)
    elif arguments.task == 'analyzeDistance':
        DistanceModel(arguments).analyzeGroundTruth()
    elif arguments.task == 'train':
        if arguments.distance:
            DistanceModel(arguments).train(arguments.numberOfExamples, checkpoint = arguments.distanceCheckpoint, restore = arguments.r)
        else:
            RecognitionModel(arguments).train(arguments.numberOfExamples, checkpoint = arguments.checkpoint, restore = arguments.r)
    elif arguments.task == 'evaluate':
        SearchModel(arguments).evaluateAccuracy()
    elif arguments.task == 'profile':
        cProfile.run('loadExamples(%d)'%(arguments.numberOfExamples))
