from spatial_transformer import *
from mixtureDensityNetwork import *
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

TESTINGFRACTION = 0.05
CONTINUOUSROUNDING = 1
ATTENTIONCANROTATE = True

[STOP,CIRCLE,LINE,RECTANGLE,LABEL] = range(5)


class StandardPrimitiveDecoder():
    def makeNetwork(self,imageRepresentation):
        # A placeholder for each target
        self.targetPlaceholder = [ (tf.placeholder(tf.int32, [None]) if t == int
                                    else tf.placeholder(tf.float32, [None]))
                                   for t,d in self.outputDimensions ]
        if not hasattr(self, 'hiddenSizes'):
            self.hiddenSizes = [None]*len(self.outputDimensions)
        if not hasattr(self, 'attentionIndices'):
            self.attentionIndices = []
        self.attentionTransforms = []

        # A prediction for each target
        self.prediction = []
        # "hard" predictions (integers or floats)
        self.hard = []
        # "soft" predictions (logits: only for categorical variables)
        self.soft = []

        # variable in the graph representing the loss of this decoder
        self.loss = []
        
        # populate the above arrays
        
        predictionInputs = [flattenImageOutput(imageRepresentation)]
        for j,(t,d) in enumerate(self.outputDimensions):
            # should we modify the image representation using a spatial transformer?
            if j in self.attentionIndices:
                theta0 = np.array([[1., 0, 0], [0, 1., 0]]).astype('float32').flatten()
                theta = tf.layers.dense(tf.concat(predictionInputs[1:],axis = 1),
                                        6,
                                        activation = tf.nn.tanh,
                                        bias_initializer=tf.constant_initializer(theta0),
                                        kernel_initializer = tf.zeros_initializer())
                if not ATTENTIONCANROTATE:
                    # force the off diagonal entries to be 0
                    theta = tf.multiply(theta, np.array([[1., 0, 1], [0, 1., 1]]).astype('float32').flatten())
                # save the transform as a field so that we can visualize it later
                self.attentionTransforms += [theta]
                # clobber the existing image input with the region that attention is focusing on
                C = int(imageRepresentation.shape[3]) # channel count
                print "c = ",C
                transformed = spatial_transformer_network(imageRepresentation,
                                                          theta,
                                                          (self.attentionSize,self.attentionSize))
                print "transfodrmed",transformed
                flat = tf.reshape(transformed,
                                  [-1, self.attentionSize*self.attentionSize*C])
                print "flat",flat
                predictionInputs[0] = flat
                
            # construct the intermediate representation, if the decoder has one
            # also pass along the transformation if we have it
            if j in self.attentionIndices:
                intermediateRepresentation = tf.concat(predictionInputs + [theta],axis = 1)
            else:
                intermediateRepresentation = tf.concat(predictionInputs,axis = 1)
            
            if self.hiddenSizes[j] != None and self.hiddenSizes[j] > 0:
                intermediateRepresentation = tf.layers.dense(intermediateRepresentation,
                                                             self.hiddenSizes[j],
                                                             activation = tf.nn.relu)
            # decoding of categorical variables
            if t == int:
                # p = prediction
                p = tf.layers.dense(intermediateRepresentation, d, activation = None)
                self.prediction.append(p)
                predictionInputs.append(tf.one_hot(self.targetPlaceholder[j], d))
                self.hard.append(tf.cast(tf.argmax(p,dimension = 1),tf.int32))
                self.soft.append(tf.nn.log_softmax(p))
                self.loss.append(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.targetPlaceholder[j],
                                                                                logits = p))
            elif t == float:
                mixtureParameters = mixtureDensityLayer(d,intermediateRepresentation,
                                                        epsilon = 0.01,
                                                        bounds = (0,MAXIMUMCOORDINATE))
                self.prediction.append(mixtureParameters)
                predictionInputs.append(tf.reshape(self.targetPlaceholder[j], [-1,1]))
                self.loss.append(-mixtureDensityLogLikelihood(mixtureParameters,
                                                              self.targetPlaceholder[j]))
                self.soft += [None]
                self.hard += [None]

        self.loss = sum(self.loss)

    def accuracyVector(self):
        '''For each example in the batch, do hard predictions match the target? ty = [None,bool]'''
        hard = [tf.equal(h,t) for h,t in zip(self.hard,self.targetPlaceholder)
                if h != None ]
        if hard != []:
            return reduce(tf.logical_and, hard)
        else:
            return True
    def placeholders(self): return self.targetPlaceholder

    @property
    def token(self): return self.__class__.token

    def beamTrace(self, session, feed, beamSize):
        originalFeed = feed
        # makes a copy of the feed
        feed = dict([(k,feed[k]) for k in feed])

        # traces is a list of tuples of (log likelihood, sequence of predictions)
        traces = [(0.0,[])]
        for j in range(len(self.outputDimensions)):
            for k in range(j):
                feed[self.targetPlaceholder[k]] = np.array([ t[1][k] for t in traces ])
            for p in originalFeed:
                feed[p] = np.repeat(originalFeed[p], len(traces), axis = 0)
            if self.outputDimensions[j][0] == int:
                soft = session.run(self.soft[j], feed_dict = feed)
                traces = [(s + coordinateScore, trace + [coordinateIndex])
                          for traceIndex,(s,trace) in enumerate(traces)
                          for coordinateIndex,coordinateScore in enumerate(soft[traceIndex]) ]
            elif self.outputDimensions[j][0] == float:
                [u,v,p] = session.run(list(self.prediction[j]), feed_dict = feed)
                traces = [(s + coordinateScore, trace + [coordinate])
                          for traceIndex,(s,trace) in enumerate(traces)
                          for coordinate, coordinateScore in
                          beamMixture(u[traceIndex],v[traceIndex],p[traceIndex],
                                      np.arange(0,MAXIMUMCOORDINATE-1,CONTINUOUSROUNDING,dtype = 'float'),
                                      beamSize)]
            traces = sorted(traces, key = lambda t: -t[0])[:beamSize]
        return traces

    def attentionSequence(self, session, feed, l):
        # what is the sequence of attention transformations when decoding line l?
        ts = self.__class__.extractTargets(l)
        # makes a copy of the feed
        feed = dict([(k,feed[k]) for k in feed])
        for t,p in zip(ts,self.targetPlaceholder): feed[p] = np.array([t])
        return session.run(self.attentionTransforms,
                           feed_dict = feed)



class CircleDecoder(StandardPrimitiveDecoder):
    token = CIRCLE
    languagePrimitive = Circle
    
    def __init__(self, imageRepresentation, continuous, attention):
        if attention > 0:
            self.attentionIndices = [1,2]
            self.attentionSize = attention
        if continuous:
            self.outputDimensions = [(float,MAXIMUMCOORDINATE)]*3 # x,y,r
            self.hiddenSizes = [None,None,None]
        else:
            self.outputDimensions = [(int,MAXIMUMCOORDINATE)]*3 # x,y,r
            self.hiddenSizes = [None, None, None]
        self.makeNetwork(imageRepresentation)
    
    def beam(self, session, feed, beamSize):
        return [(s, Circle(AbsolutePoint(x,y),r))
                for s,[x,y,r] in self.beamTrace(session, feed, beamSize)
                if x - r > 0 and y - r > 0 and x + r < MAXIMUMCOORDINATE and y + r < MAXIMUMCOORDINATE]

    @staticmethod
    def extractTargets(l):
        if l != None and isinstance(l,Circle):
            return [l.center.x,
                    l.center.y,
                    l.radius]
        return [0,0,0]

class LabelDecoder(StandardPrimitiveDecoder):
    token = LABEL
    languagePrimitive = Label
    
    def __init__(self, imageRepresentation, continuous, attention):
        if attention > 0:
            self.attentionIndices = [1,2]
            self.attentionSize = attention
        if continuous:
            self.outputDimensions = [(float,MAXIMUMCOORDINATE)]*2+[(int,len(Label.allowedLabels))] # x,y,c
            self.hiddenSizes = [None, None, None]
        else:
            self.outputDimensions = [(int,MAXIMUMCOORDINATE)]*2+[(int,len(Label.allowedLabels))] # x,y,c
            self.hiddenSizes = [None, None, None]
        self.makeNetwork(imageRepresentation)
    
    def beam(self, session, feed, beamSize):
        return [(s, Label(AbsolutePoint(x,y),Label.allowedLabels[l]))
                for s,[x,y,l] in self.beamTrace(session, feed, beamSize)
                if x > 0 and y > 0 and x < MAXIMUMCOORDINATE and y < MAXIMUMCOORDINATE ]

    @staticmethod
    def extractTargets(l):
        if l != None and isinstance(l,Label):
            return [l.p.x,
                    l.p.y,
                    Label.allowedLabels.index(l.c)]
        return [0,0,0]

class RectangleDecoder(StandardPrimitiveDecoder):
    token = RECTANGLE
    languagePrimitive = Rectangle

    def __init__(self, imageRepresentation, continuous, attention):
        if attention > 0:
            self.attentionIndices = [1,2,3]
            self.attentionSize = attention
        if continuous:
            self.outputDimensions = [(float,MAXIMUMCOORDINATE)]*4 # x,y
            self.hiddenSizes = [None]*4
        else:
            self.outputDimensions = [(int,MAXIMUMCOORDINATE)]*4 # x,y
            self.hiddenSizes = [None]*4
        self.makeNetwork(imageRepresentation)
            

    def beam(self, session, feed, beamSize):
        return [(s, Rectangle.absolute(x1,y1,x2,y2))
                for s,[x1,y1,x2,y2] in self.beamTrace(session, feed, beamSize)
                if x1 < x2 and y1 < y2 and x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0 ]

    @staticmethod
    def extractTargets(l):
        if l != None and isinstance(l,Rectangle):
            return [l.p1.x,
                    l.p1.y,
                    l.p2.x,
                    l.p2.y]
        return [0]*4

class LineDecoder(StandardPrimitiveDecoder):
    token = LINE
    languagePrimitive = Line

    def __init__(self, imageRepresentation, continuous, attention):
        if attention > 0:
            self.attentionIndices = [1,2,3,4,5]
            self.attentionSize = attention
        if continuous:
            self.outputDimensions = [(float,MAXIMUMCOORDINATE)]*4 + [(int,2)]*2 # x,y for beginning and end; arrow/-
        else:
            self.outputDimensions = [(int,MAXIMUMCOORDINATE)]*4 + [(int,2)]*2 # x,y for beginning and end; arrow/-
        self.hiddenSizes = [None,
                            32,
                            32,
                            32,
                            None,
                            None]
        self.makeNetwork(imageRepresentation)
    
    def beam(self, session, feed, beamSize):
        return [(s, Line.absolute(x1,y1,x2,y2,arrow = arrow == 1,solid = solid == 1))
                for s,[x1,y1,x2,y2,arrow,solid] in self.beamTrace(session, feed, beamSize)
                if (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) > 0 and x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0 ]

    @staticmethod
    def extractTargets(l):
        if l != None and isinstance(l,Line):
            return [l.points[0].x,l.points[0].y,l.points[1].x,l.points[1].y,
                    int(l.arrow),
                    int(l.solid)]
        return [0]*6

class StopDecoder():
    def __init__(self, imageRepresentation, continuous, attention):
        self.outputDimensions = []
        self.loss = 0.0
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
    def __init__(self, imageRepresentation, trainingPredicatePlaceholder, continuous, attention):
        self.decoders = [k(imageRepresentation,continuous,attention) for k in PrimitiveDecoder.decoderClasses]

        self.imageRepresentation = imageRepresentation
        self.prediction = tf.layers.dense(flattenImageOutput(self.imageRepresentation), len(self.decoders))
        self.hard = tf.cast(tf.argmax(self.prediction,dimension = 1),tf.int32)
        self.soft = tf.nn.log_softmax(self.prediction)
        self.targetPlaceholder = tf.placeholder(tf.int32, [None])
        self.trainingPredicatePlaceholder = trainingPredicatePlaceholder

    def loss(self):
        # the first label is for the primitive category
        ll = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.targetPlaceholder,
                                                                          logits = self.prediction))
        for decoder in self.decoders:
            decoderMask = tf.cast(tf.equal(self.targetPlaceholder, decoder.token), tf.float32)
            decoderLoss = tf.reduce_sum(tf.multiply(decoderMask,decoder.loss))
            ll += decoderLoss

        return ll

    def accuracy(self):
        a = tf.equal(self.hard,self.targetPlaceholder)
        for decoder in self.decoders:
            if decoder.token != STOP:
                vector = decoder.accuracyVector()
                if vector != True:
                    a = tf.logical_and(a,
                                       tf.logical_or(vector, tf.not_equal(self.hard,decoder.token)))
        return tf.reduce_mean(tf.cast(a, tf.float32))

    def placeholders(self):
        p = [self.targetPlaceholder]
        for d in self.decoders: p += d.placeholders()
        return p

    @staticmethod
    def extractTargets(l):
        '''Given a line of code l, what is the array of targets (int's for categorical and float's for continuous) we expect the decoder to produce?'''
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

    def attentionSequence(self, session, feed, l):
        imageRepresentation = session.run(self.imageRepresentation, feed_dict = feed)
        feed[self.imageRepresentation] = imageRepresentation
        for d in self.decoders:
            if isinstance(l,d.__class__.languagePrimitive):
                if d.attentionTransforms == []: return []
                return d.attentionSequence(session, feed, l)

class RecurrentDecoder():
    def __init__(self, imageFeatures):
        self.unit = LSTM(imageFeatures)

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

            self.decoder = PrimitiveDecoder(c1, self.trainingPredicatePlaceholder,
                                            arguments.continuous,
                                            arguments.attention)
            self.loss = self.decoder.loss()
            self.averageAccuracy = self.decoder.accuracy()

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.arguments.learningRate).minimize(self.loss)

    @property
    def checkpointPath(self):
        return "checkpoints/recognition_%s_%s_%s%s.checkpoint"%(self.arguments.architecture,
                                                              "noisy" if self.arguments.noisy else "clean",
                                                              "continuous" if self.arguments.continuous else "discrete",
                                                              ("_attention%d"%self.arguments.attention) if self.arguments.attention > 0 else '')
    
    def loadCheckpoint(self):
        with self.session.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.session, self.checkpointPath)
            
    def train(self, numberOfExamples, restore = False):
        noisyTarget,programs = loadExamples(numberOfExamples, self.arguments.trainingData)
        
        iterator = BatchIterator(10,(np.array(noisyTarget),np.array(programs)),
                                 testingFraction = TESTINGFRACTION, stringProcessor = loadImage)
        flushEverything()

        with self.session.graph.as_default():
            initializer = tf.global_variables_initializer()
            saver = tf.train.Saver()

            if not restore:
                self.session.run(initializer)
            else:
                saver.restore(self.session, self.checkpointPath)
            
            for e in range(20):
                epicLoss = []
                epicAccuracy = []
                for ts,ps in iterator.epochExamples():
                    feed = self.makeTrainingFeed(ts,ps)
                    feed[self.trainingPredicatePlaceholder] = True
                    _,l,accuracy = self.session.run([self.optimizer, self.loss, self.averageAccuracy],
                                         feed_dict = feed)
                    if len(epicAccuracy)%1000 == 0:
                        print "\t",len(epicAccuracy),l,accuracy                    
                    epicLoss.append(l)
                    epicAccuracy.append(accuracy)
                print "Epoch %d: accuracy = %f, loss = %f"%((e+1),sum(epicAccuracy)/len(epicAccuracy),sum(epicLoss)/len(epicLoss))
                testingAccuracy = []
                for ts,ps in iterator.testingExamples():
                    feed = self.makeTrainingFeed(ts,ps)
                    feed[self.trainingPredicatePlaceholder] = False
                    testingAccuracy.append(self.session.run(self.averageAccuracy, feed_dict = feed))
                print "\tTesting accuracy = %f"%(sum(testingAccuracy)/len(testingAccuracy))
                print "Saving checkpoint: %s" % saver.save(self.session, self.checkpointPath)
                flushEverything()

    def makeTrainingFeed(self, targets, programs):
        # goal, current, predictions
        gs = []
        cs = []
        ps = []
        for target, program in zip(targets, programs):
            if not self.arguments.noisy:
                target = program.draw()
            if self.arguments.randomizeOrder:
                program = Sequence(randomlyPermuteList(program.lines))
            cs += program.drawTrace()
            for j in range(len(program) + 1):
                gs.append(target)
                #cs.append(Sequence(program.lines[:j]).draw())
                l = None
                if j < len(program): l = program.lines[j]
                ps.append(self.decoder.extractTargets(l))

        gs = np.array(gs)
        cs = np.array(cs)
        ps = np.array(ps)

        if self.arguments.noisy: gs = augmentData(gs)

        if False:
            for j in range(10):
                print ps[j,:]
                showImage(np.concatenate([gs[j],cs[j]]))
        
        
        
        
        f = {self.goalPlaceholder: gs,
             self.currentPlaceholder: cs}
        for j,p in enumerate(self.decoder.placeholders()):
            f[p] = ps[:,j] #np.array([ ps[i][j] for i in range(len(ps)) ])
        return f

    def beam(self, current, goal, beamSize):
        feed = {self.currentPlaceholder: np.array([current]),
                self.goalPlaceholder: np.array([goal])}
        return sorted(self.decoder.beam(self.session, feed, beamSize), reverse = True)
    
    def attentionSequence(self, current, goal, l):
        feed = {self.currentPlaceholder: np.array([current]),
                self.goalPlaceholder: np.array([goal])}
        return self.decoder.attentionSequence(self.session, feed, l)
    
    def analyzeFailures(self, numberOfExamples):
        failures = []
        noisyTarget,programs = loadExamples(numberOfExamples, self.arguments.trainingData)
        
        iterator = BatchIterator(1,(np.array(noisyTarget),np.array(programs)),
                                 testingFraction = TESTINGFRACTION, stringProcessor = loadImage)
        with self.session.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.session, self.checkpointPath)

            totalNumberOfAttempts = 0
            for ts,ps in iterator.testingExamples():
                if len(failures) > 100: break
                
                targetProgram = ps[0]
                feed = self.makeTrainingFeed(ts,ps)
                # break the feed up into single actions
                for j in range(len(targetProgram)):
                    if len(failures) > 100: break
                    
                    singleFeed = dict([(placeholder, np.array([feed[placeholder][j,...]]))
                                       for placeholder in feed ])
                    current, goal = singleFeed[self.currentPlaceholder][0], singleFeed[self.goalPlaceholder][0]
                    target = targetProgram.lines[j]
                    if self.arguments.continuous: target = target.round(CONTINUOUSROUNDING)
                    singleFeed[self.trainingPredicatePlaceholder] = False
                    predictions = self.beam(current, goal, 100)
                    totalNumberOfAttempts += 1
                    if predictions[0][1] != target:
                        failures.append({'current': current, 'goal': goal,
                                         'target': target,
                                         'predictions': predictions})
                        print "(failure)"
                        print "\tExpected:",target
                        print "\tActually:",predictions[0][1]
                    else:
                        print "(success)"
                        if self.arguments.attention > 0:
                            attention = self.attentionSequence(current, goal, target)
                            if attention != []:
                                print attention
                                print target
                                illustration = drawAttentionSequence(goal, attention, target)
                                saveMatrixAsImage(illustration, 'attentionIllustrations/%d.png'%(totalNumberOfAttempts - len(failures)))

        # report failures
        print "%d/%d (%f%%) failure rate"%(len(failures),totalNumberOfAttempts,
                                           100*float(len(failures))/totalNumberOfAttempts)
        # compute the average rank of the failure
        ranks = [ None if not f['target'] in map(snd,f['predictions']) else map(snd,f['predictions']).index(f['target']) + 1
                  for f in failures ]
        print ranks
        print "In the frontier %d/%d"%(len([r for r in ranks if r != None ]),len(ranks))
        ranks = [r for r in ranks if r != None ]
        print ranks
        if len(ranks) > 0:
            print "Average rank: %f"%(sum(ranks)/float(len(ranks)))

        # How many failures were of each type
        print "Circle failures: %d"%(len([ None for f in failures
                                           if isinstance(f['target'],Circle)]))
        print "Line failures: %d"%(len([ None for f in failures
                                           if isinstance(f['target'],Line)]))
        print "Rectangle failures: %d"%(len([ None for f in failures
                                           if isinstance(f['target'],Rectangle)]))
        print "Label failures: %d"%(len([ None for f in failures
                                           if isinstance(f['target'],Label)]))
        print "Stop failures: %d"%(len([ None for f in failures
                                         if None == f['target']]))
            

        for j,failure in enumerate(failures):
            saveMatrixAsImage(255*failure['current'], 'failures/%d-current.png'%j)
            saveMatrixAsImage(255*failure['goal'], 'failures/%d-goal.png'%j)
            p = failure['predictions'][0][1]
            if p == None: p = []
            else: p = [p]
            p = Sequence(p).draw()
            saveMatrixAsImage(255*(p + failure['current']), 'failures/%d-predicted.png'%j)
        
        

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
                    
class DistanceModel():
    def __init__(self,arguments):
        self.arguments = arguments
        if self.arguments.continuous:
            setSnapToGrid(False)
        
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

    @property
    def checkpointPath(self):
        return "checkpoints/distance_%s_%s.checkpoint"%(self.arguments.architecture,
                                                        "noisy" if self.arguments.noisy else "clean")
    
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

    def loadCheckpoint(self):
        with self.session.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.session, self.checkpointPath)
        
    def train(self, numberOfExamples, restore = False):
        assert self.arguments.noisy
        targetImages,targetPrograms = loadExamples(numberOfExamples, self.arguments.trainingData)
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
                saver.restore(self.session, self.checkpointPath)
            for e in range(20):
                runningAverage = 0.0
                runningAverageCount = 0
                lastUpdateTime = time()
                for images,programs in iterator.epochExamples():
                    targets, current, distances = makeDistanceExamples(images, programs,
                                                                       continuous = self.arguments.continuous,
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
                print "Saving checkpoint: %s"%(saver.save(self.session, self.checkpointPath))
                flushEverything()

    def analyzeGroundTruth(self):
        self.loadCheckpoint()
        targetNames = [ "drawings/expert-%d.png"%j for j in range(100) ]
        targetImages = map(loadImage,targetNames)
        targetSequences = map(getGroundTruthParse,targetNames)

        for j in range(100):
            s = targetSequences[j]
            for m in range(3):
                sp = s
                if m > 0:
                    for _ in range(choice([1,2,3,4])):  sp = sp.mutate()
                d = self.learnedDistances(np.array([sp.draw()]),
                                          np.array([targetImages[j]]))[0]
                d1 = len(set(map(str,s.lines)) - set(map(str,sp.lines)))
                d2 = len(set(map(str,sp.lines)) - set(map(str,s.lines)))
                print "%f\t%f"%(d[0],d[1])
                if int(round(d[0])) != d1 or int(round(d[1])) != d2:
                    print "\tvs:%d\t%d"%(d1,d2)
                    showImage(targetImages[j] + sp.draw())
        

class SearchModel():
    def __init__(self,arguments):
        self.arguments = arguments
        self.recognizer = RecognitionModel(arguments)
        self.distance = DistanceModel(arguments)

        # load the networks
        self.recognizer.loadCheckpoint()
        if not self.arguments.unguided and self.arguments.noisy:
            self.distance.loadCheckpoint()

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
                    if not self.arguments.noisy:
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
        self.distance.learnedParticleDistances(targetImage, finishedPrograms)
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

            saveMatrixAsImage(n.program.draw()*255, "%s/%d.png"%(parseDirectory, j))
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
    (f,arguments,model) = a
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
    parser.add_argument('-n','--numberOfExamples', default = 100000, type = int)
    parser.add_argument('-l','--beamLength', default = 13, type = int)
    parser.add_argument('-b','--beamWidth', default = 10, type = int)
    parser.add_argument('-t','--test', default = '', type = str)
    parser.add_argument('-r', action="store_true", default=False)
    parser.add_argument('-m','--cores', default = 1, type = int)
    parser.add_argument('--noisy',action = "store_true", default = False)
    parser.add_argument('--quiet',action = "store_true", default = False)
    parser.add_argument('--distance',action = "store_true", default = False)
    parser.add_argument('--learningRate', default = 0.001, type = float)
    parser.add_argument('--architecture', default = "original", type = str)
    parser.add_argument('--continuous', action = "store_true", default = False)
    parser.add_argument('--attention', default = 0, type = int)
    parser.add_argument('--randomizeOrder', action = "store_true", default = False)

    # parameters of sequential Monte Carlo
    parser.add_argument('-T','--temperature', default = 1.0, type = float)
    parser.add_argument('--proposalCoefficient', default = 1.0, type = float)
    parser.add_argument('--parentCoefficient', action = "store_true", default = False)
    parser.add_argument('--distanceCoefficient', default = 1.0/25.0, type = float)
    parser.add_argument('--priorCoefficient', default = 0.0, type = float)
    parser.add_argument('--mistakePenalty', default = 5.0, type = float)
    parser.add_argument('--beam', action = "store_true", default = False)
    parser.add_argument('--unguided', action = "store_true", default = False)
    parser.add_argument('--noIntersections', action = "store_true", default = False)
    parser.add_argument('--showParticles', action = "store_true", default = False)

    arguments = parser.parse_args()

    arguments.trainingData = "syntheticContinuousTrainingData.tar" if arguments.continuous else "syntheticTrainingData.tar"
    
    if arguments.task == 'showSynthetic':
        print "not implemented"
    elif arguments.task == 'test':
        fs = picturesInDirectory(arguments.test)
        model = SearchModel(arguments)
        if arguments.cores == 1:
            gt = map(handleTest, [ (f,arguments,model) for f in fs ])
        else:
            gt = Pool(arguments.cores).map(handleTest, [ (f,arguments,model) for f in fs ])
        gt = [ g for g in gt if g != None ]
        print "Got a ground truth parse correct %f"%(float(len([None for g in gt if g ]))/float(len(gt)))
    
    elif arguments.task == 'visualize':
        RecognitionModel(arguments).visualizeFilters(arguments.checkpoint)
    elif arguments.task == 'analyze':
        RecognitionModel(arguments).analyzeFailures(arguments.numberOfExamples)
    elif arguments.task == 'analyzeDistance':
        DistanceModel(arguments).analyzeGroundTruth()
    elif arguments.task == 'train':
        if arguments.distance:
            DistanceModel(arguments).train(arguments.numberOfExamples, restore = arguments.r)
        else:
            RecognitionModel(arguments).train(arguments.numberOfExamples, restore = arguments.r)
    elif arguments.task == 'evaluate':
        SearchModel(arguments).evaluateAccuracy()
