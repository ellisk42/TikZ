from learnedRanking import learnToRank
from distanceMetrics import *
from fastRender import fastRender
from recognitionModel import Particle
from groundTruthParses import groundTruth
from utilities import *


import numpy as np
import os
import pickle

MODE = 'ranking' # distance

def featuresOfParticle(p, target):
    return [p.logLikelihood, p.program.logPrior(), -0.01*asymmetricBlurredDistance(target,fastRender(p.program),
                                                                              kernelSize = 7,
                                                                              factor = 1,
                                                                              invariance = 3)]

distanceTrainingData = []
trainingData = []
print groundTruth.keys()
for k in groundTruth:
    target = loadImage(k)
    print k
    parseDirectory = k[:k.index('.')] + '-parses/'

    negatives = []
    positives = []
    for p in range(100):
        pkl = parseDirectory + 'particle%d.p'%p
        if not os.path.isfile(pkl):
            break
        particle = pickle.load(open(pkl,'rb'))
        print " [+] Loaded %s"%pkl
        particle.output = None

        if set(map(str,particle.program.lines)) == groundTruth[k]:
            positives.append(particle)
        else:
            negatives.append(particle)
    print "Got %d positive examples"%(len(positives))

    if len(positives) > 0:
        if MODE == 'ranking':
            trainingData.append((np.array(map(lambda p: featuresOfParticle(p,target),positives)),
                                 np.array(map(lambda p: featuresOfParticle(p,target),negatives))))
        else:
            distanceTrainingData.append((target,
                                         [ fastRender(p.program) for p in positives ],
                                         [ fastRender(p.program) for p in negatives ]))

if MODE == 'distance':
    print "Calibrating distance function..."
    def ranks(kernelSize, factor, invariance):
        rs = []
        for t,positives,negatives in distanceTrainingData:
            positiveScores = [ asymmetricBlurredDistance(t,p,kernelSize = kernelSize,factor = factor,invariance = invariance)
                               for p in positives  ]
            negativeScores = [ asymmetricBlurredDistance(t,p,kernelSize = kernelSize,factor = factor,invariance = invariance)
                               for p  in negatives ]
            bestPositive = min(positiveScores)
            rs.append(len([ None for n in negativeScores if n <=  bestPositive ]) + 1)
        return rs

    for k in [1,3,5,7]:
        for i in [0,1,2,3]:
            for f in [0.5,1,2,5,10]:
                rs = ranks(k,f,i)
                print (k,i,f),
                print len([ None for r in rs if r < 2 ]),
                print len([ None for r in rs if r < 5+1 ]),
                print len([ None for r in rs if r < 10+1 ])
    assert False

parameters = learnToRank(trainingData)

def ranks(w):
    rs = []
    for positives, negatives in trainingData:
        bestPositive = np.dot(positives,w).max()
        negativeScores = np.dot(negatives,w)
        rs.append((negativeScores > bestPositive).sum() + 1)
    return rs

# evaluate learned parameters
rs = ranks(parameters)
print "Top 1/5/10:"
print len([r for r in rs if r == 1 ])
print len([r for r in rs if r < 6 ])
print len([r for r in rs if r < 11 ])
print

topTen = {}
for priorWeight in range(0,30):
    for distanceWeight in range(0,30):
        w = np.array([1,priorWeight/20.0,distanceWeight/200.0])
        rs = ranks(w)
        print w
        print len([r for r in rs if r < 11 ])
        topTen[tuple(w.tolist())] = len([r for r in rs if r < 11 ])

print "\n".join(map(str,list(sorted(topTen.items(),key = lambda kv: kv[1]))))
# use the weights they give us the best in the top ten
w,_ = max(topTen.items(),key = lambda kv: kv[1])
w = np.array(w)
r = ranks(w)
print r
for j in range(1,50):
    print "Top",j,":", len([x for x in r if x < j+1 ])
print "# examples:",len(r)
print "Average rank:",sum(r)/float(len(r))
