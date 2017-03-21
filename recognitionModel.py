# from matplotlib import pyplot as plot
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

def loadExamples(numberOfExamples):
    def oneHot(n):
        vector = [0.0]*10
        vector[n] = 1.0
        return vector
    images = loadImages([ "syntheticTrainingData/individualCircle-%d.png"%j
                          for j in range(numberOfExamples) ])
    programs = loadPrograms([ "syntheticTrainingData/individualCircle-%d.p"%j
                              for j in range(numberOfExamples) ])
    targetX = [oneHot(p.center.x) for p in programs ]
    targetY = [oneHot(p.center.y) for p in programs ]

    return np.array(images), np.array(targetX), np.array(targetY)


def convolutionLayer(x, w, b, strides = 1):
    x = tf.nn.conv2d(x, w, strides = [1,strides,strides,1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)
    return x

def fullyConnectedLayer(x, w, b):
    return tf.add(tf.matmul(x, w), b)

def downsample(x, k):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def makeModel(x,w,b):
    x = tf.reshape(x, [-1, 300, 300, 1])

    print x

    x = convolutionLayer(x, w['c1'], b['c1'], strides = 10)

    print x

    # decoder
    
    x = tf.reshape(x, [-1, 900])

    # now we have two separate predictions: one for the X and one for the Y
    predictionX = fullyConnectedLayer(x, w['X'], b['X'])
    predictionY = fullyConnectedLayer(x, w['Y'], b['Y'])


    print predictionX
    print predictionY

    return predictionX,predictionY

w = {
    # 10x10 window size, 3 channels in, 1 output images
    'c1': tf.Variable(tf.random_normal([10, 10, 1, 1])),

    'X': tf.Variable(tf.random_normal([900, 10])),
    'Y': tf.Variable(tf.random_normal([900, 10]))
}
b = {
    'c1': tf.Variable(tf.random_normal([1])),

    'X': tf.Variable(tf.random_normal([10])),
    'Y': tf.Variable(tf.random_normal([10]))
}

# tensor flow variable for the input images
x = tf.placeholder(tf.float32, [None, 300, 300])

# target variables have a one hot representation
# tensor flow variable for the target output (1)
t1 = tf.placeholder(tf.float32, [None,10])
# tensor flow variable for the target output (2)
t2 = tf.placeholder(tf.float32, [None,10])


predictX,predictY = makeModel(x,w,b)



loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = t1,logits = predictX))
loss += tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = t2,logits = predictY))
print loss

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

images,targetX,targetY = loadExamples(100)

print images.shape
print images[0].min(),images[0].max()
print images[0]
print targetX
print targetY



initializer = tf.initialize_all_variables()

class BatchIterator():
    def __init__(self, batchSize, tensors):
        # side-by-side shuffle of the data
        permutation = np.random.permutation(range(xs.shape[0]))
        self.tensors = [ np.array([ t[p,...] for p in permutation ]) for t in tensors ]
        self.batchSize = batchSize
        
        self.startingIndex = 0
        self.trainingSetSize = tensors[0].shape[0]
    def next(self):
        endingIndex = self.startingIndex + self.batchSize
        if endingIndex > self.trainingSetSize:
            endingIndex = self.trainingSetSize
        batch = tuple([ t[self.startingIndex:endingIndex,...] for t in self.tensors ])
        self.startingIndex = endingIndex
        if self.startingIndex == self.trainingSetSize: self.startingIndex = 0
        return batch


#iterator = BatchIterator(100,(xs,ys))
saver = tf.train.Saver()

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'test':
        with tf.Session() as s:
            saver.restore(s,"/tmp/model.checkpoint")
            px,py = s.run([predictX,predictY],feed_dict = {x: images, t1:targetX, t2:targetY})
            for j in range(5):
                print px[j],"\n",py[j]
                print ""
    else:
        with tf.Session() as s:
            s.run(initializer)
            for i in range(5000):
                _,l = s.run([optimizer, loss], feed_dict = {x: images, t1:targetX, t2:targetY})
                if i%1 == 0:
                    print i,l
                if i%100 == 0:
                    print "Saving checkpoint: %s" % saver.save(s, "/tmp/model.checkpoint")

