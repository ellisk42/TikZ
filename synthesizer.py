from fastRender import fastRender
from sketch import synthesizeProgram,parseSketchOutput
from language import *
from utilities import showImage,loadImage,saveMatrixAsImage
from recognitionModel import Particle
from groundTruthParses import groundTruthSequence

from DSL import extrapolate,sketchToDSL,renderEvaluation

import traceback
import re
import os
import argparse
import pickle
import time
from multiprocessing import Pool

class SynthesisResult():
    def __init__(self, parse, time = None, source = None, cost = None, originalDrawing = None):
        self.originalDrawing = originalDrawing
        self.parse = parse
        self.time = time
        self.source = source
        self.cost = cost

class SynthesisJob():
    def __init__(self, parse, originalDrawing):
        self.parse = parse
        self.originalDrawing = originalDrawing
    def __str__(self):
        return "%s:\n%s"%(self.originalDrawing,str(self.parse))
    def execute(self):
        startTime = time.time()
        result = synthesizeProgram(self.parse)
        elapsedTime = time.time() - startTime
        
        return SynthesisResult(parse = self.parse,
                               time = elapsedTime,
                               originalDrawing = self.originalDrawing,
                               source = result[1] if result != None else None,
                               cost = result[0] if result != None else None)
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
    for j in range(200):
        f = directory + '/particle' + str(j) + '.p'
        if not os.path.isfile(f): break
        particles.append(pickle.load(open(f,'rb')))
        print " [+] Loaded %s"%(f)

    distanceWeight = 0.1
    priorWeight = 0.0

    particles.sort(key = lambda p: p.logLikelihood - distanceWeight*p.distance + p.program.logPrior()*priorWeight,
                   reverse = True)

    return particles[:k]

# Synthesize based on the top k particles in drawings/expert*
# Just returns the jobs to synthesize these things
def expertSynthesisJobs(k):
    jobs = []
    for j in range(200):
        originalDrawing = 'drawings/expert-%d.png'%j
        particleDirectory = 'drawings/expert-%d-parses'%j
        if not os.path.exists(originalDrawing) or not os.path.exists(particleDirectory):
            continue
        
        for p in loadTopParticles(particleDirectory, k):
            jobs.append(SynthesisJob(p.sequence(), originalDrawing))

    return jobs

def synthesizeTopK(k):
    jobs = expertSynthesisJobs(k)
    # Also synthesized from the ground truth
    for k in groundTruthSequence:
        sequence = groundTruthSequence[k]
        if all([ set(map(str,sequence.lines)) != set(map(str,j.parse.lines))
                for j in jobs ]):
            jobs.append(SynthesisJob(sequence,k))
            
    results = parallelExecute(jobs)
    with open('topSynthesisResults.p','wb') as handle:
        pickle.dump(results, handle)
    print "Dumped %d results to topSynthesisResults.p"%(len(results))
        

def viewSynthesisResults(arguments):
    d = arguments.view
    if d.endswith('.p'): files = [d]
    elif d.endswith('/'): files = [ d + f for f in os.listdir(d) if f.endswith('.p') ]
    else: assert False

    latex = []

    for f in files:
        result = pickle.load(open(f,'rb'))
        print f
        print result.source
        print parseSketchOutput(result.source)
        expertIndex = re.search('(\d+)', f)
        print expertIndex.group(1)
        expertIndex = int(expertIndex.group(1))
        e = extrapolate(sketchToDSL(parseSketchOutput(result.source)))
        if not arguments.extrapolate:
            rightEntryOfTable = '''
        \\begin{minipage}{10cm}
        \\begin{verbatim}
%s
        \\end{verbatim}
\\end{minipage}
'''%(parseSketchOutput(result.source))
        else:
            rightEntryOfTable = ""
        if e != sketchToDSL(parseSketchOutput(result.source)) and arguments.extrapolate:
            print e
            image = renderEvaluation(e, exportTo = "~/projects/TikZ/extrapolations/expert-%d-extrapolation.png"%expertIndex)
            rightEntryOfTable = '\\includegraphics[width = 5cm]{../TikZ/extrapolations/expert-%d-extrapolation.png}'%expertIndex
        if rightEntryOfTable != "":
            latex.append('''
            \\begin{tabular}{ll}
    \\includegraphics[width = 5cm]{../TikZ/drawings/expert-%d.png}&
    %s
    \\end{tabular}        
            '''%(expertIndex, rightEntryOfTable))
            print

    if arguments.latex:
        latex = '%s'%("\\\\\n".join(latex))
        name = "extrapolations.tex" if arguments.extrapolate else "synthesizerOutputs.tex"
        with open('../TikZpaper/%s'%name,'w') as handle:
            handle.write(latex)
        print "Wrote output to ../TikZpaper/%s"%name

        


def synthesizeFromSequence((parse,originalDrawing,whereToPickle)):
    print parse
    startTime = time.time()
    result = synthesizeProgram(parse)
    if result == None: print "Failure to synthesize."
    else:
        result = SynthesisResult(source = result[1],
                                 cost = result[0],
                                 parse = parse,
                                 time = time.time() - startTime,
                                 originalDrawing = originalDrawing)
        pickle.dump(result,open(whereToPickle,'wb'))
    
def synthesizeGroundTruthPrograms(arguments):
    Pool(arguments.cores).map(synthesizeFromSequence,
                             [(groundTruthSequence[k],
                               k,
                               'synthesisResults/%s-synthesizerOutput.p'%(k.replace('/','-')))
                              for k in groundTruthSequence ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Synthesis of high-level code from low-level parses')
    parser.add_argument('-d', '--directory', default = None)
    parser.add_argument('-m', '--cores', default = 1, type = int)
    parser.add_argument('--view', default = None, type = str)
    parser.add_argument('--latex', default = False, action = 'store_true')
    parser.add_argument('--synthesizeTopK', default = None,type = int)
    parser.add_argument('--extrapolate', default = False, action = 'store_true')

    arguments = parser.parse_args()

    if arguments.view:
        viewSynthesisResults(arguments)
    elif arguments.synthesizeTopK != None:
        synthesizeTopK(arguments.synthesizeTopK)
    elif arguments.directory != None:
        (arguments.directory)
    else:
        synthesizeGroundTruthPrograms(arguments)
        
