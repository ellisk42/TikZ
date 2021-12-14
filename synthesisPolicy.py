import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot

import torch
import torch.nn as nn
import torch.optim as optimization
import torch.nn.functional as F
import torchvision.transforms as T


from synthesizer import *
from utilities import sampleLogMultinomial
from timeshare import *
from extrapolate import *

import time
import numpy as np
import math
import os
from pathos.multiprocessing import ProcessingPool as Pool
import re

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

        self.parameters = torch.Tensor(
            torch.randn(self.outputDimensionality, self.inputDimensionality),
            requires_grad=True,
        )

    def zeroParameters(self):
        self.parameters.data.zero_()

    def l2parameters(self):
        return (self.parameters*self.parameters).sum()

    def save(self,f):
        print(" [+] Saving model to",f)
        torch.save(self.parameters,f)
    def load(self,f):
        print(" [+] Loading model from",f)
        self.parameters = torch.load(f)

    def scoreJobs(self,jobs):
        f = torch.from_numpy(SynthesisPolicy.featureExtractor(jobs[0].parse)).float()
        f = torch.FloatTensor(f)
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
        jobs = list(results.keys())
        probabilities = self.jobProbabilities(jobs)
        t0 = sum([ results[job].time * p for job, p in zip(jobs, probabilities) ])
        TIMEOUT = 999
        minimumCost = min([ results[j].cost for j in jobs if results[j].cost != None ] + [TIMEOUT])
        if minimumCost == TIMEOUT:
            print("TIMEOUT")
            assert False
        successes = [ results[j].cost != None and results[j].cost <= minimumCost + 1 for j in jobs ]
        p0 = sum([ p for success, p in zip(successes, probabilities) if success])
        return (t0 + 1.0).log() - (p0 + 0.001).log() #t0/(p0 + 0.0001)

    def biasOptimalTime(self,results, inverseTemperature = 1):
        jobs = list(results.keys())
        TIMEOUT = 999
        minimumCost = min([ results[j].cost for j in jobs if results[j].cost != None ] + [TIMEOUT])
        if minimumCost == TIMEOUT:
            print("TIMEOUT", len(jobs))
            assert False
        scores = self.scoreJobs(jobs)
        z = lse(scores)

        logTimes = [ math.log(results[j].time) - s + z
                     for j,s in zip(jobs, scores)
                     if results[j].cost != None and results[j].cost <= minimumCost + 1 ]
        #bestTime = min(times, key = lambda t: t.data[0])
        bestTime = softMinimum(logTimes, inverseTemperature)

        return bestTime

    def deepCoderLoss(self, results):
        jobs = list(results.keys())
        TIMEOUT = 999
        minimumCost = min([ results[j].cost for j in jobs if results[j].cost != None ] + [TIMEOUT])
        if minimumCost == TIMEOUT:
            print("TIMEOUT",sequence)
            assert False

        # Find the winning program
        bestResult = min(list(results.values()),key = lambda r: r.cost if r.cost != None else TIMEOUT)
        incremental = bestResult.job.incremental
        p = bestResult.program
        depth = p.depth()
        assert depth >= 1 and depth <= 3
        reflects = False
        loops = False
        for k in p.walk():
            if isinstance(k,Loop): loops = True
            elif isinstance(k,Reflection): reflects = True
            if loops and reflects: break

        f = torch.from_numpy(SynthesisPolicy.featureExtractor(jobs[0].parse)).float()
        f = torch.FloatTensor(f)
        y = self.parameters.matmul(f)
        #z = lse([y[3],y[4],y[5]])

        return -(binary(y[0],incremental) + binary(y[1],loops) + binary(y[2],reflects))# + y[2 + depth] - z)

    def learn(self, data, L = 'expected', foldsRemaining = 0, testingData = [], numberOfIterations = 2000, regularize = 0.0):
        o = optimization.Adam([self.parameters],lr = 0.01)

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
            elif L == 'DC':
                loss = sum(self.deepCoderLoss(results) for results in data)
                testingLoss = sum(self.deepCoderLoss(results) for results in data).data[0] if testingData != [] else 0.0
            else:
                print("unknown loss function",L)
                assert False
            regularizedLoss = loss + regularize * self.l2parameters()
            o.zero_grad()
            regularizedLoss.backward()
            o.step()



            dt = (time.time() - startTime)/(60*60)
            timePerIteration = dt/s
            timePerFold = timePerIteration*numberOfIterations
            ETAthis = timePerIteration * (numberOfIterations - s)
            ETA = timePerFold * foldsRemaining + ETAthis
            if testingData != []: testingLoss = testingLoss * len(data) / len(testingData)
            if s%10 == 0:
                print("%d/%d : training loss = %.2f : testing loss = %.2f : ETA this fold = %.2f hours : ETA all folds = %.2f hours"%(s,numberOfIterations,loss.data[0],testingLoss,ETAthis,ETA))

    def reinforce(self,data):
        o = optimization.Adam([self.parameters],lr = 0.001)

        for s in range(100):
            L = sum([ R*ll for results in data for (R,ll) in [self.rollout(results,True)] ])
            L = L/len(data)
            print(L)
            print(self.parameters)
            print(self.parameters.grad)
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
        jobs = list(results.keys())
        jobLogLikelihood = {}
        for j,s in zip(jobs,self.scoreJobs(jobs)):
            jobLogLikelihood[j] = s

        history = []
        TIMEOUT = 999
        minimumCost = min([ r.cost for r in list(results.values()) if r.cost != None ] + [TIMEOUT])
        if minimumCost == TIMEOUT:
            print("TIMEOUT",sequence)
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
                (dt,nextResult) = min(list(zip(timeToFinishEachCandidate, candidates)))
                T += dt
                if results[nextResult].cost != None and results[nextResult].cost <= minimumCost + 1: return T
                finishedJobs.append(nextResult)
                for candidate, weight in zip(candidates,resourceDistribution):
                    jobProgress[candidate] += weight*dt

        if L == 'DC':
            assert not returnLogLikelihood
            f = torch.from_numpy(
                SynthesisPolicy.featureExtractor(jobs[0].parse)
            ).float()
            f = torch.FloatTensor(f)
            y = F.sigmoid(self.parameters.matmul(f))
            incrementalScore = y.data[0]
            loopScore = y.data[1]
            reflectScore = y.data[2]

            T = 0.0
            canLoop = False
            canReflect = False
            initialIncremental = incrementalScore > 0.5
            attempts = 0
            if loopScore > reflectScore:
                attemptSequence = [(False,False),(True,False),(True,True),(False,True)]
            else:
                attemptSequence = [(False,False),(False,True),(True,True),(True,False)]
            for (canLoop,canReflect) in attemptSequence:
                attempts += 1
                for d in [2,3]:#range(1,4):
                    j1 = [ j for j in jobs \
                           if j.incremental == initialIncremental \
                           and j.canLoop == canLoop \
                           and j.canReflect == canReflect \
                           and j.maximumDepth == d ]
                    assert len(j1) == 1
                    j1 = j1[0]
                    result = results[j1]
                    T += result.time
                    if result.cost != None and result.cost <= minimumCost + 1: return T
                    j2 = [ j for j in jobs \
                           if j.incremental == (not initialIncremental) \
                           and j.canLoop == canLoop \
                           and j.canReflect == canReflect \
                           and j.maximumDepth == d ]
                    assert len(j2) == 1
                    j2 = j2[0]
                    result = results[j2]
                    T += result.time
                    if result.cost != None and result.cost <= minimumCost + 1: return T

            print("Could not get minimum cost for the following problem:",minimumCost)
            for k,v in results.items():
                print(k,v.cost)
            assert False



        time = 0
        trajectoryLogProbability = 0
        while True:
            candidates = [ j
                           for j,_ in results.items()
                           if not any([ str(j) == str(o) or (results[o].cost != None and o.subsumes(j))
                                        for o in history ])]
            if candidates == []:
                print("Minimum cost",minimumCost)
                print("All of the results...")
                for j,r in sorted(iter(results.items()), key = lambda j_r: str(j_r[0])):
                    print(j,r.cost,r.time)
                print("history:")
                for h in sorted(history,key = str):
                    print(h,'\t',results[j].cost)
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

    def timeshare(self, f, optimalCost = None, globalTimeout = None, verbose=False, parse=None,
                  outputDirectory = None):
        if outputDirectory is not None:
            os.system("mkdir  -p %s"%outputDirectory)
        f = 'drawings/expert-%d.png'%f
        parse = parse or getGroundTruthParse(f)
        jobs = [ SynthesisJob(parse, f,
                              usePrior = True,
                              maximumDepth = d,
                              canLoop = l,
                              canReflect = r,
                              incremental = i)
             for d in [1,2,3]
             for i in ([True,False] if not parse.onlyOneKindOfObject() else [False])
             for l in [True,False]
             for r in [True,False] ]
        scores = [ s.data[0] for s in self.scoreJobs(jobs) ]
        tasks = [ TimeshareTask(invokeExecuteMethod, [j], s, timeout = 2*60*60) for j,s in zip(jobs, scores) ]
        bestResult = None
        resultIndex = 0
        for result in executeTimeshareTasksFairly(tasks,
                                                  dt = 5.0, # Share 5s at a time
                                                  minimumSlice = 0.25, # don't let anything run for less than a quarter second
                                                  globalTimeout = globalTimeout):
            if result.cost != None:
                # Write the program out to a file
                if outputDirectory is not None:
                    fn = "%s/program_%d.txt"%(outputDirectory,resultIndex)
                    result.exportToFile(fn)
                    print("Exported program to",fn)
                    resultIndex += 1
                if verbose:
                    print()
                    print(" [+] Found the following program:")
                    print(result.program.pretty())
                    print()
                    print()
                if bestResult == None or bestResult.cost > result.cost:
                    bestResult = result
                if result.cost <= optimalCost + 1 and globalTimeout is None: break
                for t in tasks:
                    if result.job.subsumes(t.arguments[0]): t.finished = True
        for t in tasks: t.cleanup()
        if outputDirectory is not None and bestResult is not None:
            fn = "%s/best.txt"%(outputDirectory)
            print("Exporting best program to",fn)
            bestResult.exportToFile(fn)
        return bestResult


def loadPolicyData():
    from os.path import exists
    if os.path.exists('policyTrainingData.p'):
        with open('policyTrainingData.p','rb') as handle:
            results = pickle.load(handle)
    else:
        results = []

    resultsArray = []

    legacyFixUp = False

    for j in range(100):
        drawing = 'drawings/expert-%d.png'%j
        resultsArray.append(dict([ (r.job, r) for r in results if isinstance(r,SynthesisResult) and r.job.originalDrawing == drawing ]))
        print(" [+] Got %d results for %s"%(len(resultsArray[-1]), drawing))

        # Removed those cases where we have a cost but no program This
        # bug has been fixed, but when using old data files we don't
        # want to include these
        for job, result in resultsArray[-1].items():
            if not job.incremental and result.cost != None and result.source == None:
                result.cost = None
                legacyFixUp = True

            if result.cost != None:
                newProgram = result.program.removeDeadCode()
                if newProgram.pretty() != result.program.pretty():
                    print("WARNING: detected dead code in %d"%j)
                    print(result.program.pretty())
                    result.program = newProgram
                    result.cost = result.program.totalCost()



        # Check that the subsumption trick can never cause us to not get an optimal program
        for job1, result1 in resultsArray[-1].items():
            for job2, result2 in resultsArray[-1].items():
                if job1.subsumes(job2): # job1 is more general which implies that either there is no result or it is better than the result for job2
                    if not (result1.cost == None or result2.cost == None or result1.cost <= result2.cost):
                        print(job1,'\t',result1.cost)
                        print(result1.program.pretty())
                        print(job2,'\t',result2.cost)
                        print(result2.program.pretty())
                    assert result1.cost == None or result2.cost == None or result1.cost <= result2.cost

    if legacyFixUp:
        print("")
        print(" [?] WARNING: Fixed up legacy file.")

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

    print("Looping problems:",len(iterativeProblems))
    print("Reflecting problems:",len(reflectingProblems))
    print("Deep problems:",len(deepProblems))

    iterativeScores = [ (flag, (len(x) + len(y))/float(len(parse)))
                        for parse, flag in iterativeProblems
                        for (x,y) in [parse.usedDisplacements()] ]

TIMEOUT = 10**6
def bestPossibleTime(results):
    minimumCost = min([ r.cost for r in list(results.values()) if r.cost != None ] + [TIMEOUT])
    return (min([ r.time for r in list(results.values()) if r.cost != None and r.cost <= minimumCost + 1 ] + [TIMEOUT]))
def exactTime(results):
    minimumCost = min([ r.cost for r in list(results.values()) if r.cost != None ] + [TIMEOUT])
    return (min([ r.time for j,r in results.items()
                          if j.incremental == False and j.canLoop and j.canReflect and j.maximumDepth == 3  and r.cost != None and r.cost <= minimumCost + 1] + [TIMEOUT]))
def incrementalTime(results):
    minimumCost = min([ r.cost for r in list(results.values()) if r.cost != None ] + [TIMEOUT])
    return (min([ r.time for j,r in results.items()
                          if j.incremental and j.canLoop and j.canReflect and j.maximumDepth == 3 and r.cost != None and r.cost <= minimumCost + 1] + [TIMEOUT]))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'training and evaluation of synthesis policies')
    parser.add_argument('-f', '--features',
                        choices = ['nothing','basic','fancy','basic+fancy'],
                        default = 'basic+fancy')
    parser.add_argument('-m', '--mode',
                        choices = ['expected','bias','DC+bias','DC'],
                        default = 'bias')
    parser.add_argument('--folds', default = 10, type = int)
    parser.add_argument('--regularize', default = 0, type = float)
    parser.add_argument('-s','--steps', default = 2000, type = int)
    parser.add_argument('--evaluate', default = None, type = str)
    parser.add_argument('--save', action = 'store_true',default = False)
    parser.add_argument('--load', action = 'store_true',default = False)
    parser.add_argument('--timeout', default = None, type = int)
    parser.add_argument('--extrapolate', default = None, type = str)
    parser.add_argument('--programOutputDirectory', default=None, type=str)

    arguments = parser.parse_args()
    assert arguments.extrapolate is None or arguments.evaluate is not None

    data = loadPolicyData()
    if arguments.evaluate is None:
        data = [results for results in data
                if any([r.cost != None for r in list(results.values()) ]) ]
        print("Pruned down to %d problems"%len(data))
    totalFailures = 100 - len(data)


    print("Features:",arguments.features)
    modes = arguments.mode.split('+')
    policyRollouts = {}
    # Map from problem index to which model should be used for that problem
    testingModels = {}

    for mode in modes:
        policy = []
        numberOfFolds = arguments.folds
        foldCounter = 0
        for train, test in crossValidate(data, numberOfFolds, randomSeed = 42):
            path = 'checkpoints/policy_%s_%s_%s%d_%d.pt'%(arguments.features,mode,
                                                          '' if arguments.regularize == 0 else 'regularize%f_'%arguments.regularize,
                                                          foldCounter,arguments.folds)
            foldCounter += 1
            print("Fold %d..."%foldCounter)
            model = SynthesisPolicy()
            if arguments.load:
                model.load(path)
                print(" [+] Successfully loaded model from %s"%path)
            else:
                model.learn(train,L = mode,
                            foldsRemaining = numberOfFolds - foldCounter,
                            testingData = test,
                            numberOfIterations = arguments.steps,
                            regularize = arguments.regularize)
                if arguments.save:
                    model.save(path)
            if arguments.evaluate is None:
                policy += [ model.rollout(r,L = mode) for r in test for _ in  range(10 if mode == 'expected' else 1) ]
            else:
                assert arguments.load
                for r in test:
                    testingModels[data.index(r)] = model

        policyRollouts[mode] = policy


    if arguments.evaluate is not None:
        if arguments.evaluate == "-1":
            thingsToEvaluate = list(range(100))
        else:
            thingsToEvaluate = [arguments.evaluate]

        def policyEvaluator(problemIndex):
            try:
                problemIndex = int(problemIndex)
                parse = None
            except:
                with open(problemIndex,"rb") as handle: particle = pickle.load(handle)
                parse = particle.sequence()
                try:
                    problemIndex = int(re.search("expert-(\d+)-p",problemIndex).group(1))
                except: problemIndex = None

            if problemIndex is not None:
                costs = [ r.cost for _,r in data[problemIndex].items() if r.cost != None ]
                if costs == []: return None
                bestCost = min(costs)
                model = testingModels[problemIndex]
                jobs = list(data[problemIndex].keys())
                job2w = dict(list(zip(jobs,
                                 np.exp(normalizeLogs(np.array([ s.data[0] for s in model.scoreJobs(jobs) ]))))))
                print("Best cost:",bestCost)
                print("Results:")
                for j,r in data[problemIndex].items():
                    print(j)
                    print("COST =",r.cost,"\tTIME =",r.time,"\tWEIGHT =",job2w[j])
                    print()

                theoretical = model.rollout(data[problemIndex], L = mode)
                print("Theoretical time:",theoretical)
            else:
                bestCost = 0
                model = testingModels[0] # arbitrary
                theoretical = None


            startTime = time.time()
            result = model.timeshare(problemIndex, bestCost, globalTimeout = arguments.timeout, verbose=True,
                                     parse=parse,
                                     outputDirectory=arguments.programOutputDirectory)
            actualTime = time.time() - startTime
            print("Total time:",actualTime)

            if arguments.extrapolate:
                print("Extrapolating into",arguments.extrapolate)
                exportExtrapolations([result.program], arguments.extrapolate,
                                     "drawings/expert-%d.png"%problemIndex)
            return (actualTime,theoretical)

        discrepancies = parallelMap(1, policyEvaluator,thingsToEvaluate)
        # print "DISCREPANCIES:",discrepancies
        # with open('discrepancies.p','wb') as handle:
        #     pickle.dump(discrepancies, handle)
        sys.exit(0)



    optimistic = list(map(bestPossibleTime, data))
    exact = list(map(exactTime,data))
    incremental = list(map(incrementalTime,data))

    randomModel = SynthesisPolicy()
    randomModel.zeroParameters()
    #randomPolicy = [ randomModel.rollout(r,L = mode) for r in data for _ in range(10)  ]

    modelsToCompare = [(exact,'sketch')]
    if 'DC' in policyRollouts: modelsToCompare.append((policyRollouts['DC'], 'DC'))
    modelsToCompare.append((optimistic,'oracle'))
    if 'bias' in policyRollouts: modelsToCompare.append((policyRollouts['bias'], 'learned policy (ours)'))

    bins = np.logspace(0,6,30)
    figure = plot.figure(figsize = (8,1.6))
    plot.gca().set_xlabel('time (sec)',fontsize = 9)
    for j,(ys,l) in enumerate(modelsToCompare):
        ys += [TIMEOUT]*totalFailures
        plot.subplot(1,len(modelsToCompare),1 + j)
        plot.hist(ys, bins, alpha = 0.3, label = l)
        if j == 0: plot.ylabel('frequency',fontsize = 9)

        plot.gca().set_xscale("log")
        plot.gca().set_xticks([10**e for e in range(int(round(log10(TIMEOUT) + 1))) ])
        plot.gca().set_xticklabels([ r"$10^%d$"%e if e < 6 else r"$\infty$" for e in range(int(round(log10(TIMEOUT) + 1)))  ],
                                   fontsize = 9)
        plot.gca().set_yticklabels([])
        plot.gca().set_yticks([])
        #plot.legend(fontsize = 9)
        plot.title(l,fontsize = 9)
        # Remove timeouts
        print(l,"timeouts or gives the wrong answer",len([y for y in ys if y == TIMEOUT ]),"times")
        median = np.median(ys)
        print(l," median",median)
        ys = [y for y in ys if y != TIMEOUT ]
        print(l," mean",np.mean(ys))

        print(l," : solved within a minute:",len([y for y in ys if y <= 60.0 ]))

        plot.axvline(median, color='r', linestyle='dashed', linewidth=2)
        plot.text(median * 1.5,
                  plot.gca().get_ylim()[1]*0.7,
                  'median: %ds'%(int(median)),
                  fontsize = 7)#, rotation = 90)


    #plot.plot()
    figure.text(0.5, 0.04, 'time (sec)', ha='center', va='center',
                fontsize = 9)
    plot.tight_layout()

    figureFilename = 'policyComparison_%s_%s_%d.png'%(arguments.features,arguments.mode,arguments.folds)
    plot.savefig(figureFilename)
    os.system('convert -trim %s %s'%(figureFilename,figureFilename))
    os.system('feh %s'%figureFilename)
