import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot

import torch
import torch.nn as nn
import torch.optim as optimization
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T


from synthesizer import *
from utilities import sampleLogMultinomial
from timeshare import *

import time
import numpy as np
import math
import os
from pathos.multiprocessing import ProcessingPool as Pool


def binary(x,f):
    if not f: x = -x
    return (F.sigmoid(x) + 0.0001).log()
    
def lse(xs):
    largest = xs[0].data[0]
    for x in xs:
        if x.data[0] > largest:
            largest = x.data[0]
    return largest + sum([ (x - largest).exp() for x in xs ]).log()

def softMinimum(xs, inverseTemperature):
    # returns \sum_x x * \frac{e^{-\beta*x}}{\sum_x' e^{-\beta*x'}}
    # n.b.:
    # another alternative to try is :
    # \log softMinimum = LSE{ \log x - \beta*x - LSE{-\beta*x'}}
    # This calculates the same thing but does it all in logs
    scores = [ -x*inverseTemperature for x in xs ]
    logNormalizer = lse(scores)
    logProbabilities = [ s - logNormalizer for s in scores ]
    return sum([ x * l.exp() for x,l in zip(xs,logProbabilities) ])


class SynthesisPolicy():#nn.Module):
    def __init__(self):
        self.inputDimensionality = len(SynthesisPolicy.featureExtractor(Sequence([])))
        self.outputDimensionality = 6

        self.parameters = Variable(torch.randn(self.outputDimensionality,self.inputDimensionality),
                                   requires_grad = True)

    def zeroParameters(self):
        self.parameters.data.zero_()

    def l2parameters(self):
        return (self.parameters*self.parameters).sum()

    def save(self,f):
        print " [+] Saving model to",f
        torch.save(self.parameters,f)
    def load(self,f):
        print " [+] Loading model from",f
        self.parameters = torch.load(f)

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

    def biasOptimalTime(self,results, inverseTemperature = 1):
        jobs = results.keys()
        TIMEOUT = 999
        minimumCost = min([ results[j].cost for j in jobs if results[j].cost != None ] + [TIMEOUT])
        if minimumCost == TIMEOUT:
            print "TIMEOUT",sequence
            assert False
        scores = self.scoreJobs(jobs)
        z = lse(scores)
        
        logTimes = [ math.log(results[j].time) - s + z
                     for j,s in zip(jobs, scores)
                     if results[j].cost != None and results[j].cost <= minimumCost + 1 ]
        #bestTime = min(times, key = lambda t: t.data[0])
        bestTime = softMinimum(logTimes, inverseTemperature)
        
        return bestTime

    def learn(self, data, L = 'expected', foldsRemaining = 0, testingData = [], numberOfIterations = 2000, regularize = 0.0):
        o = optimization.Adam([self.parameters],lr = 0.1)

        startTime = time.time()
        for s in range(1,numberOfIterations+1):
            if L == 'expected':
                loss = sum([self.expectedTime(results) for results in data ])
                testingLoss = sum([self.expectedTime(results) for results in testingData ]).data[0] if testingData != [] else 0.0
            elif L == 'bias':
                # anneal the inverse temperature linearly toward 2
                B = 2*float(s)/numberOfIterations
                loss = sum([self.biasOptimalTime(results, B) for results in data ])
                testingLoss = sum([self.biasOptimalTime(results, B) for results in testingData ]).data[0] if testingData != [] else 0.0
            else:
                print "unknown loss function",L
                assert False
            regularizedLoss = loss + regularize * self.l2parameters()
            o.zero_grad()
            loss.backward()
            o.step()


            
            dt = (time.time() - startTime)/(60*60)
            timePerIteration = dt/s
            timePerFold = timePerIteration*numberOfIterations
            ETAthis = timePerIteration * (numberOfIterations - s)
            ETA = timePerFold * foldsRemaining + ETAthis
            if testingData != []: testingLoss = testingLoss * len(data) / len(testingData)
            if s%100 == 0:
                print "%d/%d : training loss = %.2f : testing loss = %.2f : ETA this fold = %.2f hours : ETA all folds = %.2f hours"%(s,numberOfIterations,loss.data[0],testingLoss,ETAthis,ETA)

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
        #return np.array([len(sequence.lines),math.log(len(sequence.lines) + 1),1])
        basicFeatures = [len([x for x in sequence.lines if isinstance(x,k) ])
                         for k in [Line,Circle,Rectangle]]
        x,y = sequence.usedDisplacements()
        v = sequence.usedVectors()
        fancyFeatures = [len(x) + len(y),
                         len(sequence.usedCoordinates()),
                         frequencyOfMode(v)]
        if arguments.features == 'basic+fancy':
            return np.array(basicFeatures + fancyFeatures + [1])
        if arguments.features == 'fancy':
            return np.array(fancyFeatures + [1])
        if arguments.features == 'basic':
            return np.array(basicFeatures + [1])
        if arguments.features == 'nothing':
            return np.array([1])
        assert False
        

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
                candidates = [ j for j in jobs
                               if not any([ finished == j or \
                                            (results[finished].cost != None and finished.subsumes(j))
                                            for finished in finishedJobs ]) ]
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
                for j,r in sorted(results.iteritems(), key = lambda (j,r): str(j)):
                    print j,r.cost,r.time
                print "history:"
                for h in sorted(history,key = str):
                    print h,'\t',results[j].cost
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

    def timeshare(self, f, optimalCost = None, globalTimeout = None):
        f = 'drawings/expert-%d.png'%f
        parse = getGroundTruthParse(f)
        jobs = [ SynthesisJob(parse, f,
                              usePrior = True,
                              maximumDepth = d,
                              canLoop = l,
                              canReflect = r,
                              incremental = i)
             for d in [1,2,3]
             for i in [True,False]
             for l in [True,False]
             for r in [True,False] ]
        scores = [ s.data[0] for s in self.scoreJobs(jobs) ]
        tasks = [ TimeshareTask(invokeExecuteMethod, [j], s) for j,s in zip(jobs, scores) ]
        bestResult = None
        for result in executeTimeshareTasks(tasks,
                                            dt = 5.0, # Share 5s at a time
                                            minimumSlice = 1.0, # don't let anything run for less than a second
                                            globalTimeout = globalTimeout):
            if result.cost != None:
                if bestResult == None or bestResult.cost > result.cost:
                    bestResult = result
                if result.cost <= optimalCost + 1: break
                for t in tasks:
                    if result.job.subsumes(t.arguments[0]): t.finished = True            
        for t in tasks: t.cleanup()
        return bestResult

            
def loadPolicyData():
    with open('policyTrainingData.p','rb') as handle:
        results = pickle.load(handle)

    resultsArray = []

    legacyFixUp = False

    for j in range(100):
        drawing = 'drawings/expert-%d.png'%j
        resultsArray.append(dict([ (r.job, r) for r in results if isinstance(r,SynthesisResult) and r.job.originalDrawing == drawing ]))
        print " [+] Got %d results for %s"%(len(resultsArray[-1]), drawing)

        # Removed those cases where we have a cost but no program This
        # bug has been fixed, but when using old data files we don't
        # want to include these
        for job, result in resultsArray[-1].iteritems():
            if not job.incremental and result.cost != None and result.source == None:
                result.cost = None
                legacyFixUp = True

            if result.cost != None:
                newProgram = result.program.removeDeadCode()
                if newProgram.pretty() != result.program.pretty():
                    print "WARNING: detected dead code in %d"%j
                    print result.program.pretty()
                    result.program = newProgram
                    result.cost = result.program.totalCost()

        

        # Check that the subsumption trick can never cause us to not get an optimal program
        for job1, result1 in resultsArray[-1].iteritems():
            for job2, result2 in resultsArray[-1].iteritems():            
                if job1.subsumes(job2): # job1 is more general which implies that either there is no result or it is better than the result for job2
                    if not (result1.cost == None or result2.cost == None or result1.cost <= result2.cost):
                        print job1,'\t',result1.cost
                        print result1.program.pretty()
                        print job2,'\t',result2.cost
                        print result2.program.pretty()
                    assert result1.cost == None or result2.cost == None or result1.cost <= result2.cost
        
    if legacyFixUp:
        print ""
        print " [?] WARNING: Fixed up legacy file."

    return resultsArray

 
            


def analyzePossibleFeatures(data):
    reflectingProblems = []
    iterativeProblems = []
    deepProblems = []
    for results in data:
        for j in results:
            r = results[j]
            if r.cost != None and r.program == None:
                assert not j.incremental
                try: r.program = parseSketchOutput(results[j].source)
                except: r.cost = None
        successfulJobs = [ j for j in results if results[j].cost != None ]
        if successfulJobs == []: continue

        bestJob = min(successfulJobs, key = lambda j: results[j].cost)
        bestProgram = results[bestJob].program
        best = bestProgram.pretty()

        iterativeProblems.append((bestJob.parse,'for' in best))
        reflectingProblems.append((bestJob.parse,'reflect' in best))
        deepProblems.append((bestJob.parse,bestProgram.depth() > 2))

    print "Looping problems:",len(iterativeProblems)
    print "Reflecting problems:",len(reflectingProblems)
    print "Deep problems:",len(deepProblems)

    iterativeScores = [ (flag, (len(x) + len(y))/float(len(parse)))
                        for parse, flag in iterativeProblems
                        for (x,y) in [parse.usedDisplacements()] ]
    
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
    import argparse
    parser = argparse.ArgumentParser(description = 'training and evaluation of synthesis policies')
    parser.add_argument('-f', '--features',
                        choices = ['nothing','basic','fancy','basic+fancy'],
                        default = 'basic+fancy')
    parser.add_argument('-m', '--mode',
                        choices = ['expected','bias'],
                        default = 'bias')
    parser.add_argument('--folds', default = 10, type = int)
    parser.add_argument('--regularize', default = 0, type = float)
    parser.add_argument('-s','--steps', default = 2000, type = int)
    parser.add_argument('--evaluate', default = None, type = int)
    parser.add_argument('--save', action = 'store_true',default = False)
    parser.add_argument('--load', action = 'store_true',default = False)
    parser.add_argument('--timeout', default = None, type = int)

    arguments = parser.parse_args()

    data = loadPolicyData()
    if not arguments.evaluate:
        data = [results for results in data
                if any([r.cost != None for r in results.values() ]) ]
        print "Pruned down to %d problems"%len(data)
    totalFailures = 100 - len(data)


    print "Features:",arguments.features
    mode = arguments.mode
    policy = []
    numberOfFolds = arguments.folds
    foldCounter = 0
    for train, test in crossValidate(data, numberOfFolds, randomSeed = 42):
        path = 'checkpoints/policy_%s_%s_%s%d_%d.pt'%(arguments.features,arguments.mode,
                                                      '' if arguments.regularize == 0 else 'regularize%f_'%arguments.regularize,
                                                      foldCounter,arguments.folds)            
        foldCounter += 1
        print "Fold %d..."%foldCounter
        model = SynthesisPolicy()
        if arguments.load:
            model.load(path)
            print " [+] Successfully loaded model from %s"%path
        else:
            model.learn(train,L = mode,
                        foldsRemaining = numberOfFolds - foldCounter,
                        testingData = test,
                        numberOfIterations = arguments.steps,
                        regularize = arguments.regularize)
            if arguments.save:
                model.save(path)            
        if not arguments.evaluate:
            policy += [ model.rollout(r,L = mode) for r in test for _ in  range(10 if mode == 'expected' else 1) ]
        else:
            assert arguments.load


    if arguments.evaluate != None:
        if arguments.evaluate == -1:
            thingsToEvaluate = list(range(100))
        else: thingsToEvaluate = [arguments.evaluate]
        
        def policyEvaluator(problemIndex):
            costs = [ r.cost for _,r in data[problemIndex].iteritems() if r.cost != None ]
            if costs == []: return None
            bestCost = min(costs)
            print "Best cost:",bestCost
            print "Results:"
            for j,r in data[problemIndex].iteritems():
                print j
                print r.cost,r.time
                print 
            theoretical = model.rollout(data[problemIndex], L = mode)
            print "Theoretical time:",theoretical
            startTime = time.time()
            model.timeshare(problemIndex, bestCost, globalTimeout = arguments.timeout)
            actualTime = time.time() - startTime
            print "Total time:",actualTime
            return (actualTime,theoretical)
        
        discrepancies = Pool(10).map(policyEvaluator,thingsToEvaluate)
        print "DISCREPANCIES:",discrepancies
        with open('discrepancies.p','wb') as handle:
            pickle.dump(discrepancies, handle)
        sys.exit(0)
        
        
    
    optimistic = map(bestPossibleTime, data)
    exact = map(exactTime,data)
    incremental = map(incrementalTime,data)

    randomModel = SynthesisPolicy()
    randomModel.zeroParameters()
    #randomPolicy = [ randomModel.rollout(r,L = mode) for r in data for _ in range(10)  ]

    
    bins = np.logspace(0,5,30)
    figure = plot.figure(figsize = (6,2))
    for j,(ys,l) in enumerate([(exact,'sketch'),(optimistic,'oracle'),(policy,'learned policy (ours)')]):
        ys += [10**5]*(totalFailures*len(ys)/(100 - totalFailures))
        plot.subplot(1,3,1 + j)
        plot.hist(ys, bins, alpha = 0.3, label = l)
        if j == 1: plot.gca().set_xlabel('time (sec)',fontsize = 9)
        if j == 0: plot.ylabel('frequency',fontsize = 9)

        plot.gca().set_xscale("log")
        plot.gca().set_xticks([10**e for e in range(6) ])
        plot.gca().set_xticklabels([ r"$10^%d$"%e if e < 5 else r"$\infty$" for e in range(6)  ],
                                   fontsize = 9)
        plot.gca().set_yticklabels([])
        plot.gca().set_yticks([])
        #plot.legend(fontsize = 9)
        plot.title(l,fontsize = 9)
        # Remove timeouts
        print l,"timeouts or gives the wrong answer",len([y for y in ys if y == TIMEOUT ]),"times"
        median = np.median(ys)
        print l," median",median
        ys = [y for y in ys if y != TIMEOUT ]
        print l," mean",np.mean(ys)

        plot.axvline(median, color='r', linestyle='dashed', linewidth=2)
        plot.text(median * 1.5,
                  plot.gca().get_ylim()[1]*0.8,
                  'median: %ds'%(int(median)),
                  fontsize = 7)#, rotation = 90)
        

    plot.tight_layout()

    figureFilename = 'policyComparison_%s_%s_%d.png'%(arguments.features,arguments.mode,arguments.folds)
    plot.savefig(figureFilename)
    os.system('convert -trim %s %s'%(figureFilename,figureFilename))
    
    
    
