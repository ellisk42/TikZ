from learnedRanking import learnToRank
from similarity import analyzeFeatures
from render import render
#from fastRender import fastRender
from sketch import synthesizeProgram
from language import *
from utilities import showImage,loadImage,saveMatrixAsImage,mergeDictionaries,frameImageNicely
from recognitionModel import Particle
from groundTruthParses import groundTruthSequence,getGroundTruthParse
from extrapolate import *
from DSL import *

import traceback
import re
import os
import argparse
import pickle
import time
from pathos.multiprocessing import ProcessingPool as Pool
import matplotlib.pyplot as plot
import sys

class SynthesisResult():
    def __init__(self, job, time = None, source = None, program = None, cost = None):
        self.job = job
        self.program = program
        self.time = time
        self.source = source
        self.cost = cost
    def __str__(self):
        return "SynthesisResult(%s)"%(self.job)
    def exportToFile(self,f):
        with open(f,"w") as handle:
            handle.write("Found the following cost-%d program after %f seconds:\n%s"%
                                     (self.cost, self.time,
                                      self.program.pretty()))


class SynthesisJob():
    def __init__(self, parse, originalDrawing, usePrior = True, maximumDepth = 3, canLoop = True, canReflect = True, incremental = False):
        self.incremental = incremental
        self.maximumDepth = maximumDepth
        self.canLoop = canLoop
        self.canReflect = canReflect

        self.parse = parse
        self.originalDrawing = originalDrawing
        self.usePrior = usePrior
    def __str__(self):
        return "SynthesisJob(%s,incremental = %s,maximumD = %s,loops = %s,reflects = %s,prior = %s)"%(self.originalDrawing,
                                                     self.incremental,
                                                     self.maximumDepth,
                                                     self.canLoop,
                                                     self.canReflect,
                                                     self.usePrior)

    def subsumes(self,other):
        assert self.originalDrawing == other.originalDrawing
        if self.incremental: return False # ??? need to understand this better...
        return self.incremental == other.incremental and self.maximumDepth >= other.maximumDepth and self.canLoop >= other.canLoop and self.canReflect >= other.canReflect #and not self.incremental

    def execute(self, timeout = 60, parallelSolving = 1):
        if self.incremental: return self.executeIncrementally(timeout = timeout, parallelSolving = parallelSolving)
        else: return self.executeJoint(timeout = timeout, parallelSolving = parallelSolving)
    def executeJoint(self, timeout = 60, parallelSolving = 1):
        startTime = time.time()
        result = synthesizeProgram(self.parse,self.usePrior,
                                   maximumDepth = self.maximumDepth,
                                   canLoop = self.canLoop,
                                   canReflect = self.canReflect,
                                   CPUs = parallelSolving,
                                   timeout = timeout)
        elapsedTime = time.time() - startTime

        return SynthesisResult(self,
                               time = elapsedTime,
                               source = result[1] if result != None else None,
                               cost = result[0] if result != None else None,
                               program = parseSketchOutput(result[1]) if result != None else None)

    def executeIncrementally(self, timeout = 60, parallelSolving = 1):
        jobs = {}
        for l in self.parse.lines:
            if isinstance(l,Circle): jobs['Circle'] = jobs.get('Circle',[]) + [l]
            elif isinstance(l,Rectangle): jobs['Rectangle'] = jobs.get('Rectangle',[]) + [l]
            elif isinstance(l,Line):
                jobs['Line%s%s'%(l.solid,l.arrow)] = jobs.get('Line%s%s'%(l.solid,l.arrow),[]) + [l]
            else: assert False

        # Heuristic: try to solve the "big enough" problems first
        # Break ties by absolute size
        jobOrdering = sorted(list(jobs.keys()),key = lambda stuff: (len(stuff) < 3,len(stuff)))

        jobResults = {}
        startTime = time.time()
        xCoefficients = set([])
        yCoefficients = set([])
        usedReflections = set([])
        usedLoops = []
        for k in jobOrdering:
            print("Synthesizing for:\n",Sequence(jobs[k]))
            print("xCoefficients",xCoefficients)
            print("yCoefficients",yCoefficients)
            print("usedReflections",usedReflections)
            print("usedLoops",usedLoops)
            print("canLoop",self.canLoop)
            print("canReflect",self.canReflect)
            jobResults[k] = synthesizeProgram(Sequence(jobs[k]),
                                              self.usePrior,
                                              entireParse = self.parse,
                                              xCoefficients = xCoefficients,
                                              yCoefficients = yCoefficients,
                                              usedReflections = usedReflections,
                                              usedLoops = usedLoops,
                                              CPUs = parallelSolving,
                                              maximumDepth = self.maximumDepth,
                                              canLoop = self.canLoop,
                                              canReflect = self.canReflect,
                                              timeout = timeout)
            if jobResults[k] == None:
                print(" [-] Incremental synthesis failure: %s"%self)
                return SynthesisResult(self,
                                       time = time.time() - startTime,
                                       source = [ s[1] for s in list(jobResults.values()) if s != None ],
                                       program = None,
                                       cost = None)
            parsedOutput = parseSketchOutput(jobResults[k][1])
            xs,ys = parsedOutput.usedCoefficients()
            xCoefficients = xCoefficients|xs
            yCoefficients = yCoefficients|ys
            xr,yr = parsedOutput.usedReflections()
            usedReflections = usedReflections|set([(x,0) for x in xr ])
            usedReflections = usedReflections|set([(0,y) for y in yr ])
            usedLoops += list(parsedOutput.usedLoops())
            usedLoops = removeDuplicateStrings(usedLoops)
        elapsedTime = time.time() - startTime

        print("Optimizing using rewrites...")
        try:
            gluedTogether = Block([ x for _,result in list(jobResults.values())
                                    for x in parseSketchOutput(result).items ])
            optimalCost,optimalProgram = gluedTogether.optimizeUsingRewrites()
            print(optimalProgram.pretty())
        except:
            e = sys.exc_info()[0]
            print(" [-] Problem parsing or optimizing %s: %s"%(self.originalDrawing,e))
            optimalProgram = None
            optimalCost = None

        return SynthesisResult(self,
                               time = elapsedTime,
                               source = [ s for _,s in list(jobResults.values()) ],
                               program = optimalProgram,
                               cost = optimalCost)

def invokeExecuteMethod(k, timeout = 60, parallelSolving = 1):
    try:
        return k.execute(timeout = timeout, parallelSolving = parallelSolving)
    except Exception as exception:
        t = traceback.format_exc()
        print("Exception while executing job:\n%s\n%s\n%s\n"%(exception,t,k))
        return exception

def parallelExecute(jobs):
    if arguments.cores == 1:
        return [invokeExecuteMethod(j, timeout = arguments.timeout) for j in jobs]
    else:
        return Pool(arguments.cores).map(lambda j: invokeExecuteMethod(j,timeout = arguments.timeout),jobs)


# Loads all of the particles in the directory, up to the first 200
# Returns the top K as measured by a linear combination of image distance and neural network likelihood
def loadTopParticles(directory, k):
    particles = []
    if directory.endswith('/'): directory = directory[:-1]
    for j in range(k):
        f = directory + '/particle' + str(j) + '.p'
        if not os.path.isfile(f): break
        particles.append(pickle.load(open(f,'rb')))
        print(" [+] Loaded %s"%(f))

    return particles[:k]

# Synthesize based on the top k particles in drawings/expert*
# Just returns the jobs to synthesize these things
def expertSynthesisJobs(k):
    jobs = []
    for j in range(100):
        originalDrawing = 'drawings/expert-%d.png'%j
        particleDirectory = 'drawings/expert-%d-parses'%j
        if not os.path.exists(originalDrawing) or not os.path.exists(particleDirectory):
            continue
        newJobs = []
        for p in loadTopParticles(particleDirectory, k):
            newJobs.append(SynthesisJob(p.sequence(), originalDrawing, usePrior = not arguments.noPrior))
        # but we don't care about synthesizing if there wasn't a ground truth in them
        if any([ newJob.parse == getGroundTruthParse(originalDrawing) for newJob in newJobs ]):
            jobs += newJobs



    return jobs

def synthesizeTopK(k):
    if k == 0:
        name = 'groundTruthSynthesisResults.p'
    else:
        name = 'top%dSynthesisResults.p'%k

    jobs = expertSynthesisJobs(k) if k > 0 else []
    # synthesized from the ground truth?
    if k == 0:
        for k in groundTruthSequence:
            sequence = groundTruthSequence[k]
            if all([ not (r.parse == sequence)
                    for r in results ]):
                jobs.append(SynthesisJob(sequence,k,usePrior = True))
                if arguments.noPrior:
                    jobs.append(SynthesisJob(sequence,k,usePrior = False))
    else:
        print("top jobs",len(jobs))

    print("# jobs",len(jobs))
    flushEverything()

    results = parallelExecute(jobs) + results
    with open(name,'wb') as handle:
        pickle.dump(results, handle)
    print("Dumped %d results to %s"%(len(results),name))

def makePolicyTrainingData():
    jobs = [ SynthesisJob(getGroundTruthParse(f), f,
                          usePrior = True,
                          maximumDepth = d,
                          canLoop = l,
                          canReflect = r,
                          incremental = i)
             for j in range(100)
             for f in ['drawings/expert-%d.png'%j]
             for d in [1,2,3]
             for i in [True,False]
             for l in [True,False]
             for r in [True,False] ]
    print(" [+] Constructed %d job objects for the purpose of training a policy"%(len(jobs)))
    results = parallelExecute(jobs)
    fn = 'policyTrainingData.p'
    with open(fn,'wb') as handle:
        pickle.dump(results, handle)
    print(" [+] Dumped results to %s."%fn)




def viewSynthesisResults(arguments):
    results = pickle.load(open(arguments.name,'rb'))
    print(" [+] Loaded %d synthesis results."%(len(results)))

    interestingExtrapolations = [7,
                                 #14,
                                 17,
                                 29,
                                 #35,
                                 52,
                                 57,
                                 63,
                                 70,
                                 72,
                                 88,
                                 #99]
                                 ]
    interestingExtrapolations = [(16,12),#*
                                 #(17,0),
                                 (18,0),#*
                                 #(22,0),
                                 #(23,0),
                                 #(29,12),
                                 #(31,27),
                                 (34,0),#*
                                 #(36,0),
                                 #(38,12),
                                 (39,0),#*
                                 #(41,1),
                                 #(51,1),
                                 #(52,12),
                                 #(57,0),
                                 #(58,0),
                                 (60,0),#*
                                 #(63,0),
                                 (66,2),#*
                                 (71,1),#*
                                 #(72,0),
        #(73,0),
                                 #(74,10),
                                 #(75,5),
                                 #(79,0),
                                 #(85,1),
                                 (86,0),#*
                                 #(88,0),
                                 (90,2),#*
                                 #(92,0),
                                 #(95,8)
    ]

    #interestingExtrapolations = list(range(100))


    latex = []
    extrapolationMatrix = []

    programFeatures = {}

    for expertIndex in list(range(100)):
        f = 'drawings/expert-%d.png'%expertIndex
        parse = getGroundTruthParse(f)
        if parse == None:
            print("No ground truth for %d"%expertIndex)
            assert False

        relevantResults = [ r for r in results if r.job.originalDrawing == f and r.cost != None ]
        if relevantResults == []:
            print("No synthesis result for %s"%f)
            result = None
        else:
            result = min(relevantResults, key = lambda r: r.cost)
            equallyGoodResults = [ r for r in relevantResults if r.cost <= result.cost + 1 ]
            if len(equallyGoodResults) > 1:
                print("Got %d results for %d"%(len(equallyGoodResults),expertIndex))

            programs = [ r.program.fixStringParameters().\
                         fixReflections(result.job.parse.canonicalTranslation()).removeDeadCode()
                         for r in equallyGoodResults ]
            gt = result.job.parse.canonicalTranslation()
            badPrograms = [ p
                            for p in programs
                            if p.convertToSequence().canonicalTranslation() != gt ]
            if badPrograms:
                print(" [-] WARNING: Got %d programs that are inconsistent with ground truth"%(len(badPrograms)))
            if False:
                for program in programs:
                    prediction = program.convertToSequence().canonicalTranslation()
                    actual = gt
                    if not (prediction == actual):
                        print("FATAL: program does notproduce spec")
                        print("Specification:")
                        print(actual)
                        print("Program:")
                        print(program)
                        print(program.pretty())
                        print("Program output:")
                        print(prediction)
                        print(set(map(str,prediction.lines)))
                        print(set(map(str,actual.lines)))
                        print(set(map(str,actual.lines))^set(map(str,prediction.lines)))
                        assert False

        if result == None and arguments.extrapolate:
            print("Synthesis failure for %s"%f)
            continue

        print(" [+] %s"%f)
        print("\t(synthesis time: %s)"%(result.time if result else None))
        print()

        if arguments.debug:
            print(result.source)

        if result != None:
            syntaxTree = result.program.fixStringParameters()
            syntaxTree = syntaxTree.fixReflections(result.job.parse.canonicalTranslation())
            print(syntaxTree.pretty())
            print(syntaxTree.features())
            print(syntaxTree.convertToSequence())
            #showImage(fastRender(syntaxTree.convertToSequence()) + loadImage(f)*0.5 + fastRender(result.parse))
            programFeatures[f] = syntaxTree.features()

        if arguments.extrapolate:
            extrapolations = proposeExtrapolations(programs)
            if extrapolations:
                framedExtrapolations = [1 - frameImageNicely(loadImage(f))] + \
                                       [ frameImageNicely(t.draw(adjustCanvasSize = True))
                                         for t in extrapolations ]
                a = 255*makeImageArray(framedExtrapolations)
                extrapolationMatrix.append(a)
                print("Saving extrapolation column to",'extrapolations/expert-%d-extrapolation.png'%expertIndex)
                saveMatrixAsImage(a,'extrapolations/expert-%d-extrapolation.png'%expertIndex)



        if not arguments.extrapolate:
            rightEntryOfTable = '''
        \\begin{minipage}{10cm}
        \\begin{verbatim}
%s
        \\end{verbatim}
\\end{minipage}
'''%(syntaxTree.pretty() if result != None else "Solver timeout")
        else:
            rightEntryOfTable = ""
        if False and extrapolations != [] and arguments.extrapolate:

            #print e
            rightEntryOfTable = '\\includegraphics[width = 5cm]{../TikZ/extrapolations/expert-%d-extrapolation.png}'%expertIndex
        if rightEntryOfTable != "":
            parseImage = '\\includegraphics[width = 5cm]{../TikZ/drawings/expert-%d-parses/0.png}'%expertIndex
            if not os.path.exists('drawings/expert-%d-parses/0.png'%expertIndex):
                parseImage = "Sampled no finished traces."
            latex.append('''
            \\begin{tabular}{lll}
    \\includegraphics[width = 5cm]{../TikZ/drawings/expert-%d.png}&
            %s&
    %s
    \\end{tabular}
            '''%(expertIndex, parseImage, rightEntryOfTable))
            print()

    if arguments.latex:
        latex = '%s'%("\\\\\n".join(latex))
        name = "extrapolations.tex" if arguments.extrapolate else "synthesizerOutputs.tex"
        with open('../TikZpaper/%s'%name,'w') as handle:
            handle.write(latex)
        print("Wrote output to ../TikZpaper/%s"%name)

    if arguments.similarity:
        analyzeFeatures(programFeatures)

    if arguments.extrapolate:
        #}make the big matrix
        bigMatrix = np.zeros((max([m.shape[0] for m in extrapolationMatrix ]),256*len(extrapolationMatrix)))
        for j,r in enumerate(extrapolationMatrix):
            bigMatrix[0:r.shape[0],256*j:256*(j+1)] = r
        saveMatrixAsImage(bigMatrix,'extrapolations/allTheExtrapolations.png')


def rankUsingPrograms():
    results = pickle.load(open(arguments.name,'rb'))
    print(" [+] Loaded %d synthesis results from %s."%(len(results),arguments.name))

    def getProgramForParse(sequence):
        for r in results:
            if sequence == r.parse and r.usedPrior():
                return r
        return None

    def featuresOfParticle(p):
        r = getProgramForParse(p.sequence())
        if r != None and r.cost != None and r.source != None:
            programFeatures = mergeDictionaries({'failure': 0.0},
                                                parseSketchOutput(r.source).features())
        else:
            programFeatures = {'failure': 1.0}
        parseFeatures = {'distance': p.distance[0] + p.distance[1],
                'logPrior': p.sequence().logPrior(),
                'logLikelihood': p.logLikelihood}
        return mergeDictionaries(parseFeatures,programFeatures)

    k = arguments.learnToRank
    topParticles = [loadTopParticles('drawings/expert-%d-parses'%j,k)
                    for j in range(100) ]
    learningProblems = []
    for j,ps in enumerate(topParticles):
        gt = getGroundTruthParse('drawings/expert-%d.png'%j)
        positives = []
        negatives = []
        for p in ps:
            if p.sequence() == gt: positives.append(p)
            else: negatives.append(p)
        if positives != [] and negatives != []:
            learningProblems.append((list(map(featuresOfParticle,positives)),
                                     list(map(featuresOfParticle,negatives))))

    featureIndices = list(set([ f
                                for pn in learningProblems
                                for exs in pn
                                for ex in exs
                                for f in list(ex.keys()) ]))
    def dictionaryToVector(featureMap):
        return [ featureMap.get(f,0.0) for f in featureIndices ]

    learningProblems = [ (list(map(dictionaryToVector,positives)), list(map(dictionaryToVector,negatives)))
                         for positives,negatives in learningProblems ]
    parameters = learnToRank(learningProblems)
    for f,p in zip(featureIndices,parameters):
        print(f,p)

    # showcases where it succeeds
    programAccuracy = 0
    oldAccuracy = 0
    for j,tp in enumerate(topParticles):
        if tp == []: continue

        gt = getGroundTruthParse('drawings/expert-%d.png'%j)
        # the_top_particles_according_to_the_learned_weights
        featureVectors = np.array([ dictionaryToVector(featuresOfParticle(p))
                                    for p in tp ])
        particleScores = featureVectors.dot(parameters)
        bestParticleUsingPrograms = max(list(zip(particleScores.tolist(),tp)))[1]
        programPredictionCorrect = False
        if bestParticleUsingPrograms.sequence() == gt:
            print("Prediction using the program is correct.")
            programPredictionCorrect = True
            programAccuracy += 1
        else:
            print("Prediction using the program is incorrect.")
        oldPredictionCorrect = tp[0].sequence() == gt
        print("Was the old prediction correct?",oldPredictionCorrect)
        oldAccuracy += int(oldPredictionCorrect)

        visualization = np.zeros((256,256*3))
        visualization[:,:256] = 1 - frameImageNicely(loadImage('drawings/expert-%d.png'%j))
        visualization[:,256:(256*2)] = frameImageNicely(fastRender(tp[0].sequence()))
        visualization[:,(256*2):(256*3)] = frameImageNicely(fastRender(bestParticleUsingPrograms.sequence()))
        visualization[:,256] = 0.5
        visualization[:,256*2] = 0.5
        visualization = 255*visualization

        if not oldPredictionCorrect and programPredictionCorrect:
            fp = "../TikZpaper/figures/programSuccess%d.png"%j
            print("Great success! see %s"%fp)
            saveMatrixAsImage(visualization,fp)


        if oldPredictionCorrect and not programPredictionCorrect:
            print("Minor setback!")
            print(particleScores)




    print(programAccuracy,"vs",oldAccuracy)

def induceAbstractions():
    results = pickle.load(open(arguments.name,'rb'))
    print(" [+] Loaded %d synthesis results from %s."%(len(results),arguments.name))

    def getProgram(index):
        for r in results:
            if r.originalDrawing == 'drawings/expert-%d.png'%index:
                if r.source == None: return None
                return parseSketchOutput(r.source)
        return None

    abstractions = []
    for i in range(100):
        p1 = getProgram(i)
        if p1 == None:
            print("No synthesis result for %d"%i)
            continue

        print("Trying to induce abstractions using:")
        print(p1.pretty())
        for j in range(i+1,100):
            p2 = getProgram(j)
            if p2 == None: continue

            try:
                a,e = p1.abstract(p2,Environment())
                print("SUCCESS:")
                print(p2.pretty())
                print(a.pretty())
                abstractions.append((i,j,a,e))
            except AbstractionFailure: pass
    abstractionMatrix = []
    for i,j,a,e in abstractions:
        p = a.pretty()
        if 'for ' in p:
            print(p,"\n")
            firstProgram = a.substitute(e.firstInstantiation()).convertToSequence()
            secondProgram = a.substitute(e.secondInstantiation()).convertToSequence()
            allowUnattached = firstProgram.haveUnattachedLines() or secondProgram.haveUnattachedLines()
            samples = []
            desiredNumberOfSamples = 20
            samplingAttempts = 0
            while len(samples) < desiredNumberOfSamples and samplingAttempts < 10000:
                samplingAttempts += 1
                concrete = a.substitute(e.randomInstantiation()).convertToSequence()
                if (not concrete.hasCollisions()\
                    and (allowUnattached or (not concrete.haveUnattachedLines())))\
                    or samplingAttempts > 90000:
                    (x0,y0,_,_) = concrete.extent()
                    concrete = concrete.translate(-x0 + 1,-y0 + 1)
                    try:
                        samples.append(concrete.draw())
                    except ZeroDivisionError: pass
            samples += [np.zeros((256,256)) + 0.5]*(desiredNumberOfSamples - len(samples))
            samples = [1 - loadExpert(i),1 - loadExpert(j)] + samples
            print(firstProgram)
            print(firstProgram.haveUnattachedLines())
            print(i)
            print(secondProgram)
            print(secondProgram.haveUnattachedLines())
            print(j)
            showImage(np.concatenate([firstProgram.draw(),secondProgram.draw()],axis = 1))
            abstractionMatrix.append(np.concatenate(samples,axis = 1))
    #.showImage(np.concatenate(abstractionMatrix,axis = 0),)
    saveMatrixAsImage(255*np.concatenate(abstractionMatrix,axis = 0),'abstractions.png')


def analyzeSynthesisTime():
    results = pickle.load(open(arguments.name,'rb'))
    print(" [+] Loaded %d synthesis results from %s."%(len(results),arguments.name))
    times = []
    traceSizes = []
    programSizes = []
    for r in results:
        if not hasattr(r,'time'):
            print("missing time attribute...",r,r.__class__.__name__)
            continue

        if isinstance(r.time,list): times.append(sum(r.time))
        else: times.append(r.time)
        traceSizes.append(len(r.parse.lines))
        programSizes.append(r.cost)

    successfulResults = set([r.originalDrawing for r in results if hasattr(r,'time') ])
    print(set(['drawings/expert-%d.png'%j for j in range(100) ]) - successfulResults)

    plot.subplot(211)
    plot.title(arguments.name)
    plot.scatter([c for c,t in zip(programSizes,times) if programSizes ],
                 [t for c,t in zip(programSizes,times) if programSizes ])
    plot.xlabel('program cost')
    plot.ylabel('synthesis time in seconds')
    plot.gca().set_yscale('log')

    plot.subplot(212)
    plot.scatter(traceSizes,times)
    plot.xlabel('# of primitives in image')
    plot.ylabel('synthesis time in seconds')
    plot.gca().set_yscale('log')
    plot.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Synthesis of high-level code from low-level parses')
    parser.add_argument('-f', '--file', default = None)
    parser.add_argument('-m', '--cores', default = 1, type = int)
    parser.add_argument('--parallelSolving', default = 1, type = int)
    parser.add_argument('-n', '--name', default = "groundTruthSynthesisResults.p", type = str)
    parser.add_argument('-v', '--view', default = False, action = 'store_true')
    parser.add_argument('--latex', default = False, action = 'store_true')
    parser.add_argument('-k','--synthesizeTopK', default = None,type = int)
    parser.add_argument('-e','--extrapolate', default = False, action = 'store_true')
    parser.add_argument('--noPrior', default = False, action = 'store_true')
    parser.add_argument('--debug', default = False, action = 'store_true')
    parser.add_argument('--similarity', default = False, action = 'store_true')
    parser.add_argument('--learnToRank', default = None, type = int)
    parser.add_argument('--incremental', default = False, action = 'store_true')
    parser.add_argument('--abstract', default = False, action = 'store_true')
    parser.add_argument('--timeout', default = 60, type = int)
    parser.add_argument('--analyzeSynthesisTime', action = 'store_true')
    parser.add_argument('--makePolicyTrainingData', action = 'store_true')

    arguments = parser.parse_args()

    if arguments.view:
        viewSynthesisResults(arguments)
    elif arguments.makePolicyTrainingData:
        makePolicyTrainingData()
    elif arguments.analyzeSynthesisTime:
        analyzeSynthesisTime()
    elif arguments.learnToRank != None:
        rankUsingPrograms()
    elif arguments.abstract:
        induceAbstractions()
    elif arguments.synthesizeTopK != None:
        synthesizeTopK(arguments.synthesizeTopK)
    elif arguments.file != None:
        if "drawings/expert-%s.png"%(arguments.file) in groundTruthSequence:
            j = SynthesisJob(groundTruthSequence["drawings/expert-%s.png"%(arguments.file)],'',
                             usePrior = not arguments.noPrior,
                             incremental = arguments.incremental)
            print(j)
            s = j.execute()
            if arguments.incremental:
                print("Sketch output for each job:")
                for o in s.source:
                    print(o)
                    print(str(parseSketchOutput(o)))
                    print()
                print("Pretty printed merged output:")
                print(s.program.pretty())
            else:
                print("Parsed sketch output:")
                print(str(parseSketchOutput(s.source)))
            print(s.time,'sec')
        else:
            j = SynthesisJob(pickle.load(open(arguments.file,'rb')).program,'',
                             usePrior = not arguments.noPrior,
                             incremental = arguments.incremental)
            print(j)
            r = j.execute(timeout = arguments.timeout,parallelSolving = arguments.parallelSolving)
            print("Synthesis time:",r.time)
            print("Program:")
            print(r.program.pretty())
