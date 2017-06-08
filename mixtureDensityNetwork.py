from utilities import lseList

import math
import numpy as np
import matplotlib.pyplot as plot
import tensorflow as tf

def mixtureDensityLayer(components, inputs, epsilon = 0.0, bounds = None):
    # predicted means
    u = tf.layers.dense(inputs, components)
    if bounds != None:
        (upper, lower) = bounds
        d = upper - lower
        u = d*tf.nn.sigmoid(u) + lower
    # predicted variance
    v = tf.layers.dense(inputs, components, activation = tf.nn.softplus) + epsilon
    # mixture coefficients
    p = tf.layers.dense(inputs, components, activation = tf.nn.softmax)

    return (u,v,p)

def mixtureDensityLogLikelihood((u,v,p), target):
    #print "u = ",u
    #print "target = ",target
    components = u.shape[1]
    #print "stacked target = ",tf.stack([target]*components,axis = 1)
    
    d = u - tf.stack([target]*components,axis = 1)
    logLikelihoods = -d*d*tf.reciprocal(2.0*v) - 0.5*tf.log(v) + tf.log(p)

    return tf.reduce_logsumexp(logLikelihoods,axis = 1)

def sampleMixture(u,v,p):
    components = len(u)
    j = np.random.choice(range(components),p = p)
    return np.random.normal()*(v[j]**0.5) + u[j]

def beamMixture(u,v,p,interval,k):
    def score(x):
        return lseList([ math.log(p[j]) - 0.5*math.log(v[j]) - 0.5*(x - u[j])**2/v[j]
                         for j in range(len(u)) ])
    scores = [(i,score(i)) for i in interval ]
    return sorted(scores,key = lambda z: z[1],reverse = True)[:k]

def approximateMixtureMAP((u,v,p)):
    maximums = tf.one_hot(tf.argmax(p,axis = 1), u.shape[1])#, dtype = tf.int32)
    print u
    print maximums
    mostLikelyMean = tf.reduce_sum(u*maximums,axis = 1)


if __name__ == '__main__':
    NSAMPLE = 1000
    
    regressionInput = tf.placeholder(tf.float32, [None,1])
    regressionOutput = tf.placeholder(tf.float32, [None])

    hidden = tf.layers.dense(regressionInput, 15,
                             activation = tf.nn.sigmoid)
    
    mixtureOutput = mixtureDensityLayer(5, hidden, epsilon = 0.01, bounds = (-20,20))
    scalerPrediction = approximateMixtureMAP(mixtureOutput)
    mixtureLikelihoods = mixtureDensityLogLikelihood(mixtureOutput, regressionOutput)

    loss =  - tf.reduce_sum(mixtureLikelihoods)
    optimize = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)



    x_data = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
    r_data = np.float32(np.random.normal(size=(NSAMPLE,1)))
    y_data = np.float32(np.sin(0.75*x_data)*7.0+x_data*0.5+r_data*1.0)

    (x_data,y_data) = (y_data,x_data)

    print x_data.shape,regressionInput
    print y_data.shape,regressionOutput


    s = tf.Session()
    s.run(tf.global_variables_initializer())

    for _ in range(10000):
        feed = {regressionInput: x_data.reshape((NSAMPLE,1)),
                regressionOutput: y_data.reshape((NSAMPLE,))}
        print s.run([loss,optimize],
                    feed_dict = feed)[0]
        # print s.run(scalerPrediction, feed_dict = feed)
        # assert False


    (predictMeans,predictVariance,predictMixture) = s.run(list(mixtureOutput),
                                                          feed_dict = {regressionInput: x_data.reshape((NSAMPLE,1))})
    print predictMeans.shape
    print predictVariance.shape
    print predictMixture.shape
    print predictMeans[0]
    y_predicted = [sampleMixture(predictMeans[j],predictVariance[j],predictMixture[j])
                   for j in range(NSAMPLE) ]
    xs = np.arange(-10,10,0.1)
    #y_predicted = [beamMixture(predictMeans[j],predictVariance[j],predictMixture[j], xs, 1)[0][0]
    #for j in range(NSAMPLE) ]

    
    plot.figure(figsize=(8, 8))
    plot_out = plot.plot(x_data,y_data,'ro',alpha=0.3)
    plot_out = plot.plot(x_data,y_predicted,'bo',alpha=0.3)
    plot.show()

