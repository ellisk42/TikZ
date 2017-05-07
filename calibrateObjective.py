from recognitionModel import Particle
from groundTruthParses import groundTruth
from utilities import *


import numpy as np
import os
import pickle

def featuresOfParticle(p):
    return [p.logLikelihood, p.program.logPrior(), -p.distance]

trainingData = []
for k in groundTruth:
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
        trainingData.append((np.array(map(featuresOfParticle,positives)),
                             np.array(map(featuresOfParticle,negatives))))

def ranks(w):
    rs = []
    for positives, negatives in trainingData:
        bestPositive = np.dot(positives,w).max()
        negativeScores = np.dot(negatives,w)
        rs.append((negativeScores > bestPositive).sum() + 1)
    return rs
topTen = {}
for priorWeight in range(0,30):
    for distanceWeight in range(0,30):
        w = np.array([1,priorWeight/20.0,distanceWeight/200.0])
        rs = ranks(w)
        print w
        print len([r for r in rs if r < 11 ])
        topTen[str(w)] = len([r for r in rs if r < 11 ])

print "\n".join(map(str,list(sorted(topTen.items(),key = lambda kv: kv[1]))))
