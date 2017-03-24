from batch import BatchIterator

import matplotlib.pyplot as plot
import sys
import numpy as np
import tensorflow as tf
import os
from PIL import Image
import pickle

learning_rate = 0.001

def loadImages(filenames):
    def processPicture(p):
        p = p.convert('L')
        (w,h) = p.size
        return 1.0 - np.array(list(p.getdata())).reshape((h,w))/255.0
    return [ processPicture(Image.open(n)) for n in filenames ]

def loadPrograms(filenames):
    return [ pickle.load(open(n,'rb')) for n in filenames ]

def loadExamples(numberOfExamples, filePrefix):
    programs = loadPrograms([ "%s-%d.p"%(filePrefix,j)
                              for j in range(numberOfExamples) ])
    startingExamples = []
    endingExamples = []
    targetX = []
    targetY = []

    # get one example from each line of each program
    for j,program in enumerate(programs):
        trace = loadImages([ "%s-%d-%d.png"%(filePrefix, j, k) for k in range(len(program)) ])
        targetImage = trace[-1]
        currentImage = np.zeros(targetImage.shape)
        for k,l in enumerate(program.lines):
            x,y = l.center.x,l.center.y
            startingExamples.append(currentImage)
            endingExamples.append(targetImage)
            targetX.append(x)
            targetY.append(y)
            currentImage = trace[k]
    
    return np.array(startingExamples), np.array(endingExamples), np.array(targetX), np.array(targetY)

class RecognitionModel():
    def __init__(self):
        self.inputPlaceholder = tf.placeholder(tf.float32, [None, 300, 300, 2])
        self.targetPlaceholder1 = tf.placeholder(tf.int32, [None])
        self.targetPlaceholder2 = tf.placeholder(tf.int32, [None])

        c1 = tf.layers.conv2d(inputs = self.inputPlaceholder,
                              filters = 1,
                              kernel_size = [10,10],
                              padding = "same",
                              activation = tf.nn.relu,
                              strides = 10)

        f1 = tf.reshape(c1, [-1, 900])

        self.prediction1 = tf.layers.dense(f1, 10, activation = None)
        self.prediction2 = tf.layers.dense(f1, 10, activation = None)

        self.hard1 = tf.cast(tf.argmax(self.prediction1,dimension = 1),tf.int32)
        self.hard2 = tf.cast(tf.argmax(self.prediction2,dimension = 1),tf.int32)
        
        self.averageAccuracy = tf.reduce_mean(tf.cast(tf.logical_and(tf.equal(self.hard1,self.targetPlaceholder1),
                                                                     tf.equal(self.hard2,self.targetPlaceholder2)), tf.float32))

        self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.targetPlaceholder1,logits = self.prediction1))
        self.loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.targetPlaceholder2,logits = self.prediction2))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def train(self, numberOfExamples, exampleType, checkpoint = "/tmp/model.checkpoint"):
        partialImages,targetImages,targetX,targetY = loadExamples(numberOfExamples,
                                                                  "syntheticTrainingData/"+exampleType)
        images = np.stack([partialImages,targetImages],3)

        initializer = tf.global_variables_initializer()
        iterator = BatchIterator(50,(images, targetX, targetY))
        saver = tf.train.Saver()

        with tf.Session() as s:
            s.run(initializer)
            for i in range(1000):
                xs,t1s,t2s = iterator.next()
                
                _,l,accuracy = s.run([self.optimizer, self.loss, self.averageAccuracy],
                                     feed_dict = {self.inputPlaceholder: xs,
                                                  self.targetPlaceholder1:t1s,
                                                  self.targetPlaceholder2:t2s})
                if i%50 == 0:
                    print i,accuracy,l
                if i%100 == 0:
                    print "Saving checkpoint: %s" % saver.save(s, checkpoint)

    def test(self, numberOfExamples, exampleType, checkpoint = "/tmp/model.checkpoint"):
        partialImages,targetImages,targetX,targetY = loadExamples(numberOfExamples,
                                                                  "syntheticTrainingData/"+exampleType)
        images = np.stack([partialImages,targetImages],3)

        initializer = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as s:
            saver.restore(s,checkpoint)
            px,py,accuracy = s.run([self.hard1,self.hard2,self.averageAccuracy],
                                   feed_dict = {self.inputPlaceholder: images,
                                                self.targetPlaceholder1:targetX,
                                                self.targetPlaceholder2:targetY})
            print "Average accuracy:",accuracy
            for j in range(5):
                plot.imshow(partialImages[j],cmap = 'gray')
                plot.show()
                plot.imshow(targetImages[j],cmap = 'gray')
                plot.show()
                print px[j],py[j]
                print targetX[j],targetY[j]
                print ""

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'test':
        RecognitionModel().test(100, "doubleCircle")
    else:
        RecognitionModel().train(1000, "tripleCircle")
