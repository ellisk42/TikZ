from utilities import sampleLogMultinomial

import numpy as np
from random import choice
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.python.ops import array_ops


def sampleSequence():
    if choice([True,False]):
        s = [choice([2,1])]
        while len(s) < L - 1:
            s.append(1 + (1 - (s[-1] - 1)))
        return s + [0]
    else:
        return [choice([2,1])]*(L/2) + [0]
    
class RecurrentNetwork():
    def __init__(self, numberOfUnits, dictionarySize, maximumLength, inputFeatures = None):
        self.model = rnn.LSTMCell(numberOfUnits)
        
        self.loadingMatrix = tf.Variable(tf.random_uniform([numberOfUnits,dictionarySize],-1.0,1.0))

        self.lengthPlaceholder = tf.placeholder(tf.int32, shape = [None],name = 'LENGTH')

        self.maximumLength = maximumLength
        self.dictionarySize = dictionarySize

        if inputFeatures != None:
            transformedInputFeatures = [ tf.layers.dense(inputs = inputFeatures,
                                                         units = s,
                                                         activation = tf.nn.tanh)
                                         for s in self.model.state_size ]
            transformedInputFeatures = rnn.LSTMStateTuple(*transformedInputFeatures)
        else: transformedInputFeatures = None

        # Unrolls some number of steps L
        self.inputPlaceholder = tf.placeholder(tf.int32, shape = [None,maximumLength],name = 'INPUT')
        self.embeddedInputs = tf.nn.embedding_lookup(tf.transpose(self.loadingMatrix),self.inputPlaceholder)
        self.outputs, self.states = tf.nn.dynamic_rnn(self.model,
                                                      inputs = self.embeddedInputs,
                                                      dtype = tf.float32,
                                                      sequence_length = self.lengthPlaceholder,
                                                      initial_state = transformedInputFeatures)
        # projectedOutputs: None x timeSteps x dictionarySize
        projectedOutputs = tf.tensordot(self.outputs, self.loadingMatrix, axes = [[2],[0]])
        self.outputDistribution = tf.nn.log_softmax(projectedOutputs)
        self.hardOutputs = tf.cast(tf.argmax(projectedOutputs,dimension = 2),tf.int32)


    # sequence prediction model with prediction fed into input
    def decodesIntoLoss(self, labels):
        l = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels,
                                                           logits = self.outputDistribution)
        # l:labels
        # l:None,L
        # reduce across each time step
        l = tf.reduce_sum(l,axis = -1)
        # reduce across each example
        return tf.reduce_mean(l)

    def decodesIntoAccuracy(self, labels, perSymbol = True):
        # as the dimensions None x L
        accuracyMatrix = tf.equal(self.hardOutputs, labels)

        # zero out anything past the labeled length
        accuracyMatrix = tf.logical_and(accuracyMatrix,
                                        tf.sequence_mask(self.lengthPlaceholder, maxlen = self.maximumLength))

        # Some across all of the time steps to get the total number of predictions correct in each batch entry
        accuracyVector = tf.reduce_sum(tf.cast(accuracyMatrix,tf.int32),axis = 1)
        if perSymbol:
            # Now normalize it by the sequence length and take the average
            accuracyVector = tf.divide(tf.cast(accuracyVector,tf.float32),
                                       tf.cast(self.lengthPlaceholder,tf.float32))
        if not perSymbol:
            # accuracy is measured per sequence
            accuracyVector = tf.cast(tf.equal(accuracyVector,self.lengthPlaceholder),tf.float32)
        return tf.reduce_mean(accuracyVector)
    
    def decodingTrainingFeed(self, sequences, labels = None):
        # batch size
        B = len(sequences)

        feed = {}
        # sequence length
        feed[self.lengthPlaceholder] = np.array([len(s) for s in sequences ])
        
        # the first input is just a dummy zero entry
        i = np.zeros((B,self.maximumLength))
        for b in range(B):
            i[b,1:(len(sequences[b]) - 0)] = np.array(sequences[b][:-1])

        feed[self.inputPlaceholder] = i

        if labels != None:
            l = np.zeros((B,self.maximumLength))
            for b in range(B):
                l[b,0:len(sequences[b])] = np.array(sequences[b])
            feed[labels] = l


        return feed
    
    def sample(self, session, stopSymbol = None, baseFeed = None):
        sequenceSoFar = []
        for j in range(self.maximumLength):
            s = [0] + sequenceSoFar
            s += [0]*(self.maximumLength - len(s))
            feed = dict(baseFeed) if baseFeed != None else {}
            feed[self.lengthPlaceholder] = np.array([len(sequenceSoFar) + 1])
            feed[self.inputPlaceholder] = np.array([s])
            distribution = session.run(self.outputDistribution, feed)
            outputDistribution = distribution[0,j,:]
            n = sampleLogMultinomial(outputDistribution)
            sequenceSoFar.append(n)
            if n == stopSymbol: break
        return sequenceSoFar

    def beam(self, session, k, stopSymbol = None, baseFeed = None):
        particles = [(0.0,[])] #(log likelihood,sequence)
        finishedParticles = []
        for j in range(self.maximumLength):
            B = len(particles)
            feed = dict([ (key,np.tile(baseFeed[key],(B,1))) for key in baseFeed ]) if baseFeed != None else {}
            feed[self.lengthPlaceholder] = np.array([len(s) + 1 for _,s in particles ])
            i = np.zeros((B,self.maximumLength))
            for b in range(B):
                i[b,1:len(particles[b][1])+1] = np.array(particles[b][1])
            feed[self.inputPlaceholder] = i
            distribution = session.run(self.outputDistribution, feed)[:,j,:]

            particles = [ (particles[b][0] + distribution[b,w], particles[b][1] + [w])
                          for b in range(B)
                          for w in range(self.dictionarySize) ]
            if stopSymbol != None:
                finishedParticles += [ p for p in particles if p[1][-1] == stopSymbol ]
                particles = [ p for p in particles if p[1][-1] != stopSymbol ]
            elif j == self.maximumLength - 1:
                finishedParticles = particles

            particles = sorted(particles,reverse = True)[:k]

        return sorted(finishedParticles,reverse = True)
            
        


if __name__ == '__main__':
    VOCABULARYSIZE = 3
    L = 8
    hint = tf.placeholder(tf.float32, shape = [None,1],name = 'HINT')        
    m = RecurrentNetwork(5,VOCABULARYSIZE,L)#,hint)
    labels = tf.placeholder(tf.int32, shape = [None,L],name = 'LABELS')

    accuracy = m.decodesIntoAccuracy(labels,False)

    loss = m.decodesIntoLoss(labels)
    Optimizer = tf.train.AdamOptimizer(learning_rate = 10**-4).minimize(loss)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for i in range(20000):
            sequences = [sampleSequence() for _ in range(100) ]
            hints = np.array([ s[0] for s in sequences ])
            feed = m.decodingTrainingFeed(sequences, labels)
            feed[hint] = hints.reshape((len(sequences),1))
            l,a,_ = session.run([loss,accuracy,Optimizer],
                              feed)
            if i%1000 == 0:
                print i,l,a
        for h in [1.0,2.0]:
            print "Sampling h = %s"%h
            samples = [ tuple(m.sample(session,
                                       stopSymbol = 0,
                                       baseFeed = {hint: np.array([[h]])}))
                        for _ in range(1000)]
            histogram = {}
            for s in samples: histogram[s] = histogram.get(s,0) + 1
            histogram = sorted([(histogram[s],s) for s in histogram ])
            print "\n".join(map(str,histogram[-10:]))


            print "Beaming h = %s"%h
            b = m.beam(session, k = 3, stopSymbol = 0,
                       baseFeed = {hint: np.array([h])})
            print "\n".join(map(str,b))
