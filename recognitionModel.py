from batch import BatchIterator

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


def makeModel(x):
    x = tf.reshape(x, [-1, 300, 300, 2])

    print x

    x = tf.layers.conv2d(inputs = x,
                         filters = 1,
                         kernel_size = [10,10],
                         padding = "same",
                         activation = tf.nn.relu,
                         strides = 10)

    print x

    # decoder
    
    x = tf.reshape(x, [-1, 900])

    # now we have two separate predictions: one for the X and one for the Y
    predictionX = tf.layers.dense(x, 10, activation = None)
    predictionY = tf.layers.dense(x, 10, activation = None)


    print predictionX
    print predictionY

    return predictionX,predictionY


# tensor flow variable for the input images
x = tf.placeholder(tf.float32, [None, 300, 300, 2])

# target variables have a one hot representation
# tensor flow variable for the target output (1)
t1 = tf.placeholder(tf.int32, [None])
# tensor flow variable for the target output (2)
t2 = tf.placeholder(tf.int32, [None])


predictX,predictY = makeModel(x)
hardX,hardY = tf.argmax(predictX,dimension = 1),tf.argmax(predictY,dimension = 1)
print hardX,hardY


loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = t1,logits = predictX))
loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = t2,logits = predictY))


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

partialImages,images,targetX,targetY = loadExamples(100,"syntheticTrainingData/doubleCircle")
images = np.stack([partialImages,images],3)
print images.shape
print images[0].min(),images[0].max()


initializer = tf.global_variables_initializer()



iterator = BatchIterator(50,(images, targetX, targetY))
saver = tf.train.Saver()

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'test':
        with tf.Session() as s:
            saver.restore(s,"/tmp/model.checkpoint")
            px,py = s.run([hardX,hardY],feed_dict = {x: images})
            for j in range(10):
                print px[j],"\n",py[j]
                print ""
    else:
        with tf.Session() as s:
            s.run(initializer)
            for i in range(1000):
                xs,t1s,t2s = iterator.next()
                
                _,l = s.run([optimizer, loss], feed_dict = {x: xs, t1:t1s, t2:t2s})
                if i%50 == 0:
                    print i,l
                if i%100 == 0:
                    print "Saving checkpoint: %s" % saver.save(s, "/tmp/model.checkpoint")

