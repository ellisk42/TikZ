import tensorflow as tf
import numpy as np


def learnToRank(examples):
    '''
    example should be a list of tuples
    The first element of the tuple are the feature vectors of positive examples
    The second element of the tuple are the feature vectors of negative examples
    Learns to rank at least one of the positives above all of the negatives
    '''

    d = len(examples[0][0][0])

    w = tf.Variable(tf.random_normal([d], stddev = 0.3), name = "W") #placeholder(tf.float32,[d])
    wp = tf.reshape(w,[d,1])

    loss = 0
    for positives,negatives in examples:
        positiveScores = tf.transpose(tf.matmul(np.array(positives,dtype = np.float32),wp))
        negativeScores = tf.transpose(tf.matmul(np.array(negatives,dtype = np.float32),wp))
        print positiveScores
        print negativeScores
        scores = tf.concat([positiveScores,negativeScores],axis = 1)
        print scores

        smallNumber = 0.0000001
        maximumPositive = tf.reduce_logsumexp(positiveScores)
        maximumOverall = tf.reduce_logsumexp(scores)
        
        loss += maximumOverall - maximumPositive

    print loss

    learning_rate = 0.0001
    Optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        for j in range(1000**2):
            l,_,parameters = s.run([loss,Optimizer,w])
            if j%1000 == 0:
                print j,l,parameters
    return parameters
        
if __name__ == '__main__':
    learnToRank([([[1,2],[0,4]],
                  [[2,4],[9,3]]),
                 ([[2,9],[1,2]],
                  [[3,0]])])
