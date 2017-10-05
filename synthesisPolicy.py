from synthesizer import *
from utilities import sampleLogMultinomial

import numpy as np


import torch
import torch.nn as nn
import torch.optim as optimization
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

#def torchlse(#stuff):
    
def binary(x,f):
    if not f: x = -x
    return (F.sigmoid(x) + 0.0001).log()
    
def lse(xs):
    largest = xs[0].data[0]
    for x in xs:
        if x.data[0] > largest:
            largest = x.data[0]
    return largest + sum([ (x - largest).exp() for x in xs ]).log()

class SynthesisPolicy():#nn.Module):
    def __init__(self):
        # super(SynthesisPolicy,self).__init__()
        
        self.inputDimensionality = len(SynthesisPolicy.featureExtractor(Sequence([])))
        self.outputDimensionality = 6
        self.B = 100

        self.parameters = Variable(torch.randn(self.outputDimensionality,self.inputDimensionality),
                                   requires_grad = True)

    def scoreJobs(self,jobs):
        f = torch.from_numpy(SynthesisPolicy.featureExtractor(jobs[0].parse)).float()
        f = Variable(f)
        y = self.parameters.matmul(f)
        z = lse([y[3],y[4],y[5]])
        scores = []
        for j in jobs:
            score = binary(y[0], j.incremental) + binary(y[1], j.canLoop) + binary(y[2], j.canReflect)
            score += y[2 + j.maximumDepth] - z
            scores.append(score)
        return scores

    def jobProbabilities(self,jobs):
        scores = self.scoreJobs(jobs)
        z = lse(scores)
        return [ (score - z).exp() for score in scores ]

    def expectedTime(self,results):
        jobs = results.keys()
        probabilities = self.jobProbabilities(jobs)
        t0 = sum([ results[job].time * p for job, p in zip(jobs, probabilities) ])
        TIMEOUT = 999
        minimumCost = min([ results[j].cost for j in jobs if results[j].cost != None ] + [TIMEOUT])
        if minimumCost == TIMEOUT:
            print "TIMEOUT",sequence
            assert False
        successes = [ results[j].cost <= minimumCost + 1 for j in jobs ]
        p0 = sum([ p for success, p in zip(successes, probabilities) if success])
        return (t0 + 1.0).log() - (p0 + 0.001).log()

    def learn(self,data):
        data = [results for results in data
                if any([r.cost != None for r in results.values() ])]
        o = optimization.Adam([self.parameters],lr = 0.01)

        for s in range(100):
            loss = sum([self.expectedTime(results) for results in data ])
            print loss
            print self.parameters
            print self.parameters.grad
            loss.backward()
            o.step()
        
            

    @staticmethod
    def featureExtractor(sequence):
        return np.array([len([x for x in sequence.lines if isinstance(x,k) ])
                         for k in [Line,Circle,Rectangle]] + [1])


        

    def rollout(self, results):
        jobs = results.keys()
        jobLogLikelihood = {}#dict([ zip(jobs, self.scoreJobs(jobs)) ])
        for j,s in zip(jobs,self.scoreJobs(jobs)):
            jobLogLikelihood[j] = s.data[0]
        
        history = []
        TIMEOUT = 999
        minimumCost = min([ r.cost for r in results.values() if r.cost != None ] + [TIMEOUT])
        if minimumCost == TIMEOUT:
            print "TIMEOUT",sequence
            assert False

        time = 0
        trajectoryLogProbability = 0
        while True:
            candidates = [ j
                           for j,_ in results.iteritems()
                           if not any([ str(j) == str(o) #or (results[o].cost != None and o.subsumes(j))
                                        for o in history ])]
            if candidates == []:
                for j,r in results.iteritems():
                    print j,r.cost,r.time
                assert False
            job = candidates[sampleLogMultinomial([ jobLogLikelihood[j] for j in candidates ])]
            sample = results[job]
            time += sample.time
            history.append(job)

            if sample.cost != None and sample.cost <= minimumCost + 1:

                return time

            
def loadPolicyData():
    with open('policyTrainingData.p','rb') as handle:
        results = pickle.load(handle)

    resultsArray = []

    for j in range(100):
        drawing = 'drawings/expert-%d.png'%j
        resultsArray.append(dict([ (r.job, r) for r in results if isinstance(r,SynthesisResult) and r.job.originalDrawing == drawing ]))
        print " [+] Got %d results for %s"%(len(resultsArray[-1]), drawing)

    return resultsArray

 
            

def evaluatePolicy(results, policy):
    jobs = results.keys()
    minimumCost = min([ r.cost for r in results.values() if r.cost != None ])
    scores = map(policy, jobs)
    orderedJobs = sorted(zip(scores, jobs), reverse = True)
    print map(lambda oj: str(snd(oj)),orderedJobs)
    events = []
    T = 0.0
    minimumCostSoFar = float('inf')
    for j, (score, job) in enumerate(orderedJobs):
        if any([ o.subsumes(job) for _,o in orderedJobs[:j] ]): continue

        T += results[job].time

        if results[job].cost == None: continue
        
        normalizedCost = minimumCost/float(results[job].cost)

        if normalizedCost < minimumCostSoFar:
            minimumCostSoFar = normalizedCost
            events.append((T,normalizedCost))
    return events

TIMEOUT = 10*60*60
def bestPossibleTime(results):
    minimumCost = min([ r.cost for r in results.values() if r.cost != None ] + [TIMEOUT])
    return math.log(min([ r.time for r in results.values() if r.cost != None and r.cost <= minimumCost + 1 ] + [TIMEOUT]))
def exactTime(results):
    return math.log(min([ r.time for j,r in results.iteritems()
                          if j.incremental == False and j.canLoop and j.canReflect and j.maximumDepth == 3] + [TIMEOUT]))
def incrementalTime(results):
    return math.log(min([ r.time for j,r in results.iteritems()
                          if j.incremental and j.canLoop and j.canReflect and j.maximumDepth == 3] + [TIMEOUT]))
        
if __name__ == '__main__':
    data = loadPolicyData()
    data = [results for results in data
            if any([r.cost != None for r in results.values() ])]

    policy = SynthesisPolicy()
    policy.learn(data)
    
    policy = [policy.rollout(r) for r in data for _ in range(10) ]
    optimistic = map(bestPossibleTime,data)
    exact = map(exactTime,data)
    incremental = map(incrementalTime,data)

    print exact

    import matplotlib.pyplot as plot
    import numpy as np
    
    bins = np.linspace(0,20,40)
    for ys,l in [(exact,'exact'),(optimistic,'optimistic'),(incremental,'incremental'),(policy,'learned policy')] :
        plot.hist(ys, bins, alpha = 0.3, label = l)
    plot.legend()
    plot.show()
    
