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
    


class SynthesisPolicy():
    def __init__(self):
        self.inputDimensionality = len(SynthesisPolicy.featureExtractor(Sequence([])))
        self.outputDimensionality = 6
        self.B = 100

        self.x = Variable(torch.randn(self.B,self.inputDimensionality))

        # Due to torch stupidity, some of these are logits and some of these are not
        self.incrementalPrediction = nn.Sequential(nn.Linear(self.inputDimensionality, 1), nn.Sigmoid())
        self.loopPrediction = nn.Sequential(nn.Linear(self.inputDimensionality, 1), nn.Sigmoid())
        self.reflectPrediction = nn.Sequential(nn.Linear(self.inputDimensionality, 1), nn.Sigmoid())
        # ATTENTION: this is a logit
        self.depthPrediction = nn.Sequential(nn.Linear(self.inputDimensionality, 3), nn.LogSoftmax())

    @staticmethod
    def featureExtractor(sequence):
        return np.array([len([x for x in sequence.lines if isinstance(x,k) ])
                for k in [Line,Circle,Rectangle]])

    def rollout(self, sequence, results):
        f = SynthesisPolicy.featureExtractor(sequence)
        f = f.reshape([1,-1])
        f = Variable(torch.from_numpy(f).float())

        i = self.incrementalPrediction(f)
        l = self.loopPrediction(f)
        r = self.reflectPrediction(f)
        d = self.depthPrediction(f)

        jobLogLikelihood = {}
        for j,result in results.iteritems():        
            jobLogLikelihood[j] = \
                - nn.BCELoss()(i, Variable(torch.from_numpy(np.array([[int(j.incremental)]])).float())) - \
                nn.BCELoss()(l, Variable(torch.from_numpy(np.array([[int(j.canLoop)]])).float())) - \
                nn.BCELoss()(r, Variable(torch.from_numpy(np.array([[int(j.canReflect)]])).float())) - \
                nn.NLLLoss()(d, Variable(torch.from_numpy(np.array([int(j.maximumDepth - 1)]))))

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
                           if not any([ o.subsumes(j) for o in history ])]
            job = candidates[sampleLogMultinomial([ jobLogLikelihood[j].data for j in candidates ])]
            sample = results[job]
            time += sample.time
            
            trajectoryLogProbability += jobLogLikelihood[job]
            Z = torchlse([ jobLogLikelihood[k] for k in candidates ])
            trajectoryLogProbability -= Z

            if sample.cost != None and sample.cost <= minimumCost + 1:
                return time, trajectoryLogProbability
            
                
            
        

        # Sample a job
        i_ = torch.bernoulli(torch.exp(i.data))
        l_ = torch.bernoulli(torch.exp(l.data))
        r_ = torch.bernoulli(torch.exp(r.data))
        d_ = torch.multinomial(torch.exp(d.data),1) + 1
        
        
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

    SynthesisPolicy().rollout(data[38].keys()[0].parse,
                              data[38])


    optimistic = map(bestPossibleTime,data)
    exact = map(exactTime,data)
    incremental = map(incrementalTime,data)

    print exact

    import matplotlib.pyplot as plot
    import numpy as np
    
    bins = np.linspace(0,20,40)
    for ys,l in [(exact,'exact'),(optimistic,'optimistic'),(incremental,'incremental')] :
        plot.hist(ys, bins, alpha = 0.3, label = l)
    plot.legend()
    plot.show()
    
    for j,r in data.iteritems():
        if r.cost  == None: continue

        print j
        print r.cost
        print int(r.time/60.0),'m'
        print
    print evaluatePolicy(data, lambda j: int(j.incremental)) #featureExtractor(j.parse))
