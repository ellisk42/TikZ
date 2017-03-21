# from matplotlib import pyplot as plot
import numpy as np
import tensorflow as tf
import os


learning_rate = 0.1

def loadPictures(filenames):
    q = tf.train.string_input_producer(filenames)
    reader = tf.WholeFileReader()
    _, image_file = reader.read(q)
    image = tf.image.decode_png(image_file,channels = 1)
    pictures = []
    with tf.Session() as session:
        tf.initialize_all_variables().run()

        coordinate = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coordinate)

        for _ in range(len(filenames)):
            image_tensor = ((255.0 - session.run([image])[0])/255.0).reshape((100,100))
            pictures.append(image_tensor)        

        coordinate.request_stop()
        coordinate.join(threads)

    print "Loaded",len(filenames),"pictures."
    return pictures




def loadTrainingData():
    lines = loadPictures([ "generateOutput/lines/%d.png" % j for j in range(1000) ])
    rectangles = loadPictures(["generateOutput/rectangles/%d.png" % j for j in range(1000) ])
    circles = loadPictures(["generateOutput/circles/%d.png" % j for j in range(1000) ])
    xs = lines + rectangles + circles
    ys = [[1,0,0]]*len(lines) + [[0,1,0]]*len(rectangles) + [[0,0,1]]*len(circles)

    print "xs = ",np.array(xs)
    return np.array(xs),np.array(ys)




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


def makeModel(x, w, b):
    x = tf.reshape(x, [-1, 100, 100, 1])
#    x = downsample(x, 3) # x is now 100x100

    x = convolutionLayer(x, w['c1'], b['c1'])

    # x: [None,100,100,20]
    x = tf.reduce_sum(x, [1,2])

    x = tf.reshape(x, [-1, 20])

    x = fullyConnectedLayer(x, w['f1'], b['f1'])
    return tf.sigmoid(x)
    
    # x = downsample(x, 10)

    # x = convolutionLayer(x, w['c2'], b['c2'])
    # x = downsample(x, 10)

    # x = tf.reshape(x, [-1, 45])

    # x = fullyConnectedLayer(x, w['f1'], b['f1'])
    
    # return tf.sigmoid(x)
    

w = {
    # 10x10 window size, 3 channels in, 20 output images
    'c1': tf.Variable(tf.random_normal([10, 10, 1, 20])),
#    'c2': tf.Variable(tf.random_normal([5, 5, 20, 5])),
    'f1': tf.Variable(tf.random_normal([20, 3]))
}
b = {
    'c1': tf.Variable(tf.random_normal([20])),
#    'c2': tf.Variable(tf.random_normal([5])),
    'f1': tf.Variable(tf.random_normal([3]))
}

x = tf.placeholder(tf.float32, [None, 100, 100])
y = tf.placeholder(tf.float32, [None, 3])
predict = makeModel(x,w,b)

# calculate log likelihood
epsilon = 0.0001
loss = tf.mul(y, tf.log(predict + epsilon))
loss += tf.mul(1 - y, tf.log(1 - predict + epsilon))
loss = - tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

xs,ys = loadTrainingData()
print xs[0]
# plot.imshow(xs[2009], cmap = 'Greys')
# plot.show()
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


iterator = BatchIterator(100,(xs,ys))
saver = tf.train.Saver()

with tf.Session() as s:
    s.run(initializer)
    for i in range(5000):
        bx,by = iterator.next()
        _,l = s.run([optimizer, loss], feed_dict = {x: bx, y: by})
        if i%1 == 0:
            print i,l
        if i%100 == 0:
            print "Saving checkpoint: %s" % saver.save(s, "/tmp/model.checkpoint")
            
