from learnedRanking import learnToRank
from similarity import analyzeFeatures
from render import render
from fastRender import fastRender
from sketch import synthesizeProgram
from language import *
from utilities import showImage,loadImage,saveMatrixAsImage,mergeDictionaries,frameImageNicely
from recognitionModel import Particle
from groundTruthParses import groundTruthSequence,getGroundTruthParse

from DSL import *

import traceback
import re
import os
import argparse
import pickle
import time
from multiprocessing import Pool

class SynthesisResult():
    def __init__(self, parse, time = None, source = None, cost = None, originalDrawing = None, usePrior = True):
        self.usePrior = usePrior
        self.originalDrawing = originalDrawing
        self.parse = parse
        self.time = time
        self.source = source
        self.cost = cost
    def __str__(self):
        return self.source
    def usedPrior(self):
        if hasattr(self,'usePrior'): return self.usePrior
        return True

icingResult = SynthesisResult(getGroundTruthParse('drawings/expert-38.png'),
                              source = icingModelOutput,
                              cost = 0,
                              originalDrawing = 'drawings/expert-38.png')

class SynthesisJob():
    def __init__(self, parse, originalDrawing, usePrior = True):
        self.parse = parse
        self.originalDrawing = originalDrawing
        self.usePrior = usePrior
    def __str__(self):
        return "SynthesisJob(%s) using the prior? %s:\n%s"%(self.originalDrawing,self.usePrior,str(self.parse))
    def execute(self):
        startTime = time.time()
        result = synthesizeProgram(self.parse,self.usePrior)
        elapsedTime = time.time() - startTime
        
        return SynthesisResult(parse = self.parse,
                               time = elapsedTime,
                               originalDrawing = self.originalDrawing,
                               source = result[1] if result != None else None,
                               cost = result[0] if result != None else None,
                               usePrior = self.usePrior)
def invokeExecuteMethod(k):
    try:
        return k.execute()
    except Exception as exception:
        t = traceback.format_exc()
        print "Exception while executing job:\n%s\n%s\n%s\n"%(exception,t,k)
        return exception

def parallelExecute(jobs):
    if arguments.cores == 1:
        return map(invokeExecuteMethod,jobs)
    else:
        return Pool(arguments.cores).map(invokeExecuteMethod,jobs)


# Loads all of the particles in the directory, up to the first 200
# Returns the top K as measured by a linear combination of image distance and neural network likelihood
def loadTopParticles(directory, k):
    particles = []
    if directory.endswith('/'): directory = directory[:-1]
    for j in range(k):
        f = directory + '/particle' + str(j) + '.p'
        if not os.path.isfile(f): break
        particles.append(pickle.load(open(f,'rb')))
        print " [+] Loaded %s"%(f)

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
    if arguments.resume:
        with open(name,'rb') as handle:
            results = [ r for r in pickle.load(handle) if r.source != None ]
        print "Resuming with",len(results),"old results."
    else: results = []
    
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
        print "top jobs",len(jobs)
                    
    results = parallelExecute(jobs) + results
    with open(name,'wb') as handle:
        pickle.dump(results, handle)
    print "Dumped %d results to %s"%(len(results),name)
        

def viewSynthesisResults(arguments):
    results = pickle.load(open(arguments.name,'rb'))
    print " [+] Loaded %d synthesis results."%(len(results))

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
    interestingExtrapolations = list(range(100))
                                 

    latex = []
    extrapolationMatrix = []

    programFeatures = {}

    for expertIndex in range(100):

        if arguments.extrapolate:
            if not any([ e == expertIndex or isinstance(e,tuple) and e[0] == expertIndex
                         for e in interestingExtrapolations ]):
                continue
            
        
        f = 'drawings/expert-%d.png'%expertIndex
        parse = getGroundTruthParse(f)
        if parse == None:
            print "No ground truth for %d"%expertIndex
            continue
        parts = set(map(str,parse.lines))
        result = None
        for r in results:
            if isinstance(r,SynthesisResult) and r.originalDrawing == f:
                if set(map(str,r.parse.lines)) == parts and r.usedPrior() == (not arguments.noPrior):
                    result = r
                    break
        if j == 38: result = icingResult
        if result == None:
            print "No synthesis result for %s"%f
            if arguments.extrapolate: continue

        if result.source == None:
            print "Synthesis failure for %s"%f
            if arguments.extrapolate: continue

        print " [+] %s"%f
        print

        if arguments.debug:
            print result.source

        if result != None and result.source != None:
            syntaxTree = parseSketchOutput(result.source)
            syntaxTree = syntaxTree.fixReflections(result.parse.canonicalTranslation())
            print syntaxTree
            print syntaxTree.features()
            print syntaxTree.convertToPython()
            print syntaxTree.convertToSequence()
            #showImage(fastRender(syntaxTree.convertToSequence()) + loadImage(f)*0.5 + fastRender(result.parse))
            programFeatures[f] = syntaxTree.features()

        extrapolations = []
        if arguments.extrapolate:
#            syntaxTree = syntaxTree.explode()
            trace = syntaxTree.convertToSequence()
            print trace
            originalHasCollisions = result.parse.hasCollisions()
            print "COLLISIONS",originalHasCollisions

            framedExtrapolations = []
            for e in syntaxTree.extrapolations():
                t = e.convertToSequence()
                if not originalHasCollisions and t.removeDuplicates().hasCollisions(): continue
                if t == trace: continue
                if any([t == o for o in extrapolations ]): continue
                extrapolations.append(t)

                framedExtrapolations.append(1 - frameImageNicely(1 - t.framedRendering(result.parse)))

                if len(interestingExtrapolations) != 100:
                    break
                
            if framedExtrapolations != []:
                if arguments.debug:
                    framedExtrapolations = [loadImage(f), fastRender(syntaxTree.convertToSequence())] + framedExtrapolations
                else:
                    framedExtrapolations = [frameImageNicely(loadImage(f))] + framedExtrapolations
                a = np.zeros((256*len(framedExtrapolations),256))
                for j,e in enumerate(framedExtrapolations):
                    a[j*256:(1+j)*256,:] = 1 - e
                    a[j*256,:] = 0.5
                    a[(1+j)*256-1,:] = 0.5
                a[0,:] = 0.5
                a[:,0] = 0.5
                a[:,255] = 0.5
                a[256*len(framedExtrapolations)-1,:] = 0.5
                a = 255*a
                # to show the first one
                a = a[:(256*2),:]
                extrapolationMatrix.append(a)
                saveMatrixAsImage(a,'extrapolations/expert-%d-extrapolation.png'%expertIndex)

            
        
        if not arguments.extrapolate:
            rightEntryOfTable = '''
        \\begin{minipage}{10cm}
        \\begin{verbatim}
%s
        \\end{verbatim}
\\end{minipage}
'''%(parseSketchOutput(result.source).pretty() if result != None and result.source != None else "Solver timeout")
        else:
            rightEntryOfTable = ""
        if extrapolations != [] and arguments.extrapolate:
            #}make the big matrix
            bigMatrix = np.zeros((max([m.shape[0] for m in extrapolationMatrix ]),256*len(extrapolationMatrix)))
            for j,r in enumerate(extrapolationMatrix):
                bigMatrix[0:r.shape[0],256*j:256*(j+1)] = r
            if len(extrapolations) < 10 and False:
                saveMatrixAsImage(bigMatrix,'../TikZpaper/figures/extrapolationMatrix.png')
            else:
                saveMatrixAsImage(bigMatrix,'../TikZpaper/figures/extrapolationMatrixSupplement.png')
            print e
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
            print

    if arguments.latex:
        latex = '%s'%("\\\\\n".join(latex))
        name = "extrapolations.tex" if arguments.extrapolate else "synthesizerOutputs.tex"
        with open('../TikZpaper/%s'%name,'w') as handle:
            handle.write(latex)
        print "Wrote output to ../TikZpaper/%s"%name

    if arguments.similarity:
        analyzeFeatures(programFeatures)

        
def rankUsingPrograms():
    results = pickle.load(open(arguments.name,'rb'))
    print " [+] Loaded %d synthesis results from %s."%(len(results),arguments.name)

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
            learningProblems.append((map(featuresOfParticle,positives),
                                     map(featuresOfParticle,negatives)))

    featureIndices = list(set([ f
                                for pn in learningProblems
                                for exs in pn
                                for ex in exs 
                                for f in ex.keys() ]))
    def dictionaryToVector(featureMap):
        return [ featureMap.get(f,0.0) for f in featureIndices ]

    learningProblems = [ (map(dictionaryToVector,positives), map(dictionaryToVector,negatives))
                         for positives,negatives in learningProblems ]
    parameters = learnToRank(learningProblems)
    for f,p in zip(featureIndices,parameters):
        print f,p

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
        bestParticleUsingPrograms = max(zip(particleScores.tolist(),tp))[1]
        programPredictionCorrect = False
        if bestParticleUsingPrograms.sequence() == gt:
            print "Prediction using the program is correct."
            programPredictionCorrect = True
            programAccuracy += 1
        else:
            print "Prediction using the program is incorrect."
        oldPredictionCorrect = tp[0].sequence() == gt
        print "Was the old prediction correct?",oldPredictionCorrect
        oldAccuracy += int(oldPredictionCorrect)

        visualization = np.zeros((256,256*3))
        visualization[:,:256] = 1 - frameImageNicely(loadImage('drawings/expert-%d.png'%j))
        visualization[:,256:(256*2)] = frameImageNicely(fastRender(tp[0].sequence()))
        visualization[:,(256*2):(256*3)] = frameImageNicely(fastRender(bestParticleUsingPrograms.sequence()))
        visualization[:,256] = 0.5
        visualization[:,256*2] = 0.5
        visualization = 255*visualization
        
        if not oldPredictionCorrect and programPredictionCorrect:
            print "Great success!"
            saveMatrixAsImage(visualization,"../TikZpaper/figures/programSuccess%d.png"%j)

            
        if oldPredictionCorrect and not programPredictionCorrect:
            print "Minor setback!"
            print particleScores
            

            
            
    print programAccuracy,"vs",oldAccuracy
            
        
                                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Synthesis of high-level code from low-level parses')
    parser.add_argument('-f', '--file', default = None)
    parser.add_argument('-m', '--cores', default = 1, type = int)
    parser.add_argument('-n', '--name', default = "groundTruthSynthesisResults.p", type = str)
    parser.add_argument('--view', default = False, action = 'store_true')
    parser.add_argument('--latex', default = False, action = 'store_true')
    parser.add_argument('--synthesizeTopK', default = None,type = int)
    parser.add_argument('--extrapolate', default = False, action = 'store_true')
    parser.add_argument('--noPrior', default = False, action = 'store_true')
    parser.add_argument('--debug', default = False, action = 'store_true')
    parser.add_argument('--resume', default = False, action = 'store_true')
    parser.add_argument('--similarity', default = False, action = 'store_true')
    parser.add_argument('--learnToRank', default = None, type = int)
    

    arguments = parser.parse_args()

    if arguments.view:
        viewSynthesisResults(arguments)
    elif arguments.learnToRank != None:
        rankUsingPrograms()
    elif arguments.synthesizeTopK != None:
        synthesizeTopK(arguments.synthesizeTopK)
    elif arguments.file != None:
        if "drawings/expert-%s.png"%(arguments.file) in groundTruthSequence:
            j = SynthesisJob(groundTruthSequence["drawings/expert-%s.png"%(arguments.file)],'',
                             usePrior = not arguments.noPrior)
            print j
            s = j.execute()
            print s
            print parseSketchOutput(s.source)
        else:
            j = SynthesisJob(pickle.load(open(arguments.file,'rb')).program,'',
                             usePrior = not arguments.noPrior)
            print j
            print j.execute()
                     
