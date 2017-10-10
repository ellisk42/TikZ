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
        self.inputDimensionality = len(SynthesisPolicy.featureExtractor(Sequence([])))
        self.outputDimensionality = 6

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
        successes = [ results[j].cost != None and results[j].cost <= minimumCost + 1 for j in jobs ]
        p0 = sum([ p for success, p in zip(successes, probabilities) if success])
        return (t0 + 1.0).log() - (p0 + 0.001).log() #t0/(p0 + 0.0001)

    def biasOptimalTime(self,results):
        jobs = results.keys()
        TIMEOUT = 999
        minimumCost = min([ results[j].cost for j in jobs if results[j].cost != None ] + [TIMEOUT])
        if minimumCost == TIMEOUT:
            print "TIMEOUT",sequence
            assert False
        scores = self.scoreJobs(jobs)
        z = lse(scores)
        times = [ math.log(results[j].time) - s + z
                  for j,s in zip(jobs, scores)
                  if results[j].cost != None and results[j].cost <= minimumCost + 1 ]
        bestTime = min(times, key = lambda t: t.data[0])
        
        return bestTime.exp()

    def learn(self, data, L = 'expected'):
        o = optimization.Adam([self.parameters],lr = 0.01)

        for s in range(100):
            if L == 'expected':
                loss = sum([self.expectedTime(results) for results in data ])
            elif L == 'bias':
                loss = sum([self.biasOptimalTime(results) for results in data ])
            else:
                print "unknown loss function",L
                assert False
            print loss
            print self.parameters
            print self.parameters.grad
            o.zero_grad()
            loss.backward()
            o.step()

    def reinforce(self,data):
        o = optimization.Adam([self.parameters],lr = 0.001)

        for s in range(100):
            L = sum([ R*ll for results in data for (R,ll) in [self.rollout(results,True)] ])
            L = L/len(data)
            print L
            print self.parameters
            print self.parameters.grad
            o.zero_grad()
            L.backward()
            o.step()
        
        
            

    @staticmethod
    def featureExtractor(sequence):
        #return np.array([len(sequence.lines),1])
        return np.array([len([x for x in sequence.lines if isinstance(x,k) ])
                         for k in [Line,Circle,Rectangle]] + [1])


        

    def rollout(self, results, returnLogLikelihood = False, L = 'expected'):
        jobs = results.keys()
        jobLogLikelihood = {}
        for j,s in zip(jobs,self.scoreJobs(jobs)):
            jobLogLikelihood[j] = s
        
        history = []
        TIMEOUT = 999
        minimumCost = min([ r.cost for r in results.values() if r.cost != None ] + [TIMEOUT])
        if minimumCost == TIMEOUT:
            print "TIMEOUT",sequence
            assert False

        if L == 'bias':
            finishedJobs = []
            jobProgress = dict([(j,0.0) for j in jobs ])
            T = 0.0
            while True:
                candidates = [ j for j in jobs if not j in finishedJobs ]
                z = lse([ jobLogLikelihood[j] for j in candidates ]).data[0]
                resourceDistribution = [ math.exp(jobLogLikelihood[j].data[0] - z) for j in candidates ]
                timeToFinishEachCandidate = [ (results[j].time - jobProgress[j])/w
                                              for w,j in zip(resourceDistribution,candidates) ]
                (dt,nextResult) = min(zip(timeToFinishEachCandidate, candidates))
                T += dt
                if results[nextResult].cost != None and results[nextResult].cost <= minimumCost + 1: return T
                finishedJobs.append(nextResult)
                for candidate, weight in zip(candidates,resourceDistribution):
                    jobProgress[candidate] += weight*dt
                
                

        time = 0
        trajectoryLogProbability = 0
        while True:
            candidates = [ j
                           for j,_ in results.iteritems()
                           if not any([ str(j) == str(o) or (results[o].cost != None and o.subsumes(j))
                                        for o in history ])]
            if candidates == []:
                print "Minimum cost",minimumCost
                print "All of the results..."
                for j,r in results.iteritems():
                    print j,r.cost,r.time
                print "history:"
                for h in history:
                    print h
                assert False
            job = candidates[sampleLogMultinomial([ jobLogLikelihood[j].data[0] for j in candidates ])]
            sample = results[job]
            time += sample.time
            history.append(job)

            if returnLogLikelihood:
                trajectoryLogProbability = jobLogLikelihood[job] + trajectoryLogProbability
                z = lse([ jobLogLikelihood[k] for k in candidates ])
                trajectoryLogProbability = trajectoryLogProbability - z

            if sample.cost != None and sample.cost <= minimumCost + 1:
                if returnLogLikelihood:
                    return time, trajectoryLogProbability
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

TIMEOUT = 10**5
def bestPossibleTime(results):
    minimumCost = min([ r.cost for r in results.values() if r.cost != None ] + [TIMEOUT])
    return (min([ r.time for r in results.values() if r.cost != None and r.cost <= minimumCost + 1 ] + [TIMEOUT]))
def exactTime(results):
    minimumCost = min([ r.cost for r in results.values() if r.cost != None ] + [TIMEOUT])
    return (min([ r.time for j,r in results.iteritems()
                          if j.incremental == False and j.canLoop and j.canReflect and j.maximumDepth == 3  and r.cost != None and r.cost <= minimumCost + 1] + [TIMEOUT]))
def incrementalTime(results):
    minimumCost = min([ r.cost for r in results.values() if r.cost != None ] + [TIMEOUT])
    return (min([ r.time for j,r in results.iteritems()
                          if j.incremental and j.canLoop and j.canReflect and j.maximumDepth == 3 and r.cost != None and r.cost <= minimumCost + 1] + [TIMEOUT]))
        
if __name__ == '__main__':
    data = loadPolicyData()
    data = [results for results in data
            if any([r.cost != None for r in results.values() ]) and not '60xy' in results.keys()[0].originalDrawing]
    print "Pruned down to %d problems"%len(data)

    policy = []
    for train, test in crossValidate(data, 2):
        model = SynthesisPolicy()
        model.learn(train,L = 'expected')
        policy += [ model.rollout(r,L = 'expected') for r in test for _ in  range(10) ]
        
    
    optimistic = map(bestPossibleTime, data)*10
    exact = map(exactTime,data)*10
    incremental = map(incrementalTime,data)*10

   
    import matplotlib.pyplot as plot
    import numpy as np
    
    bins = np.logspace(0,5,30)
    plot.figure()
    for j,(ys,l) in enumerate([(exact,'exact'),(optimistic,'oracle'),(incremental,'incremental'),(policy,'learned policy')]):
        plot.subplot(int('22' + str(j + 1)))
        plot.hist(ys, bins, alpha = 0.3, label = l)
        plot.gca().set_xscale("log")
        plot.legend()
        plot.xlabel('time (sec)')
        plot.ylabel('frequency')
        # Remove timeouts
        print l,"timeouts or gives the wrong answer",len([y for y in ys if y == TIMEOUT ]),"times"
        ys = [y for y in ys if y != TIMEOUT ]
        print l," median",np.median(ys)
        print l," mean",np.mean(ys)

    plot.show()
    
