import numpy as np
from utilities import *
import tensorflow as tf
import numpy as np


def learnToRank(examples, folds = 0):
    '''
    example should be a list of tuples
    The first element of the tuple are the feature vectors of positive examples
    The second element of the tuple are the feature vectors of negative examples
    Learns to rank at least one of the positives above all of the negatives
    '''

    if folds != 0:
        if folds < 0: folds = len(examples)
        examples = np.random.permutation(examples).tolist()

        averageRanks = []
        topOne = []
        topFive = []
        topTen = []
        
        for fold in range(folds):
            testingIndexes = range(len(examples)*fold/folds,
                                   len(examples)*(fold+1)/folds)
            trainingData = [ e for j,e in enumerate(examples) if not j in testingIndexes ]
            testingData = [ e for j,e in enumerate(examples) if j in testingIndexes ]
            w = learnToRank(trainingData)
            rs = []
            for positives, negatives in testingData:
                bestPositive = np.array(positives,dtype = np.float32).dot(w).max()
                negativeScores = np.array(negatives,dtype = np.float32).dot(w).max()
                rs.append((negativeScores > bestPositive).sum() + 1)

            averageRanks.append(sum(rs)/float(len(rs)))
            topOne.append(float(len([ r for r in rs if r == 1 ]))/len(rs))
            topFive.append(float(len([ r for r in rs if r < 6 ]))/len(rs))
            topTen.append(float(len([ r for r in rs if r < 11 ]))/len(rs))

        print "Average ranks for each of the folds:",meanAndStandardError(averageRanks)
        print averageRanks
        print "Top one accuracy:",meanAndStandardError(topOne)
        print topOne
        print "Top-five accuracy:",meanAndStandardError(topFive)
        print topFive
        print "Top ten accuracy:",meanAndStandardError(topTen)
        print topTen
        return 

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

        maximumPositive = tf.reduce_logsumexp(positiveScores)
        maximumOverall = tf.reduce_logsumexp(scores)
        
        loss += maximumOverall - maximumPositive

    print loss

    learning_rate = 0.001
    Optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        for j in range(100):
            l,_,parameters = s.run([loss,Optimizer,w])
            if j%1000 == 0:
                print j,l,parameters
    return parameters
        
if __name__ == '__main__':
    learnToRank([([[1,2],[0,4]],
                  [[2,4],[9,3]]),
                 ([[2,9],[1,2]],
                  [[3,0]])]*3,
                -1)
