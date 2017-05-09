from fastRender import fastRender
from sketch import synthesizeProgram,parseSketchOutput
from language import *
from utilities import showImage,loadImage
from recognitionModel import Particle
from groundTruthParses import groundTruthSequence

import os
import argparse
import pickle
import time
from multiprocessing import Pool

class SynthesisResult():
    def __init__(self, parse, time = None, source = None, cost = None):
        self.parse = parse
        self.time = time
        self.source = source
        self.cost = cost

def viewSynthesisResults(d):
    if d.endswith('.p'): files = [d]
    elif d.endswith('/'): files = [ d + f for f in os.listdir(d) if f.endswith('.p') ]
    else: assert False

    for f in files:
        result = pickle.load(open(f,'rb'))
        print f
        print source
        print parseSketchOutput(source)
        print 

        
def loadParses(directory):
    particles = []
    if directory.endswith('/'): directory = directory[:-1]
    for j in range(100):
        f = directory + '/particle' + str(j) + '.p'
        if not os.path.isfile(f): break
        particles.append(pickle.load(open(f,'rb')))
        print " [+] Loaded %s"%(f)

    distanceWeight = 0.1
    priorWeight = 0.0

    particles.sort(key = lambda p: p.logLikelihood - distanceWeight*p.distance + p.program.logPrior()*priorWeight,
                   reverse = True)

    for p in particles[:5]:
        if not p.finished():
            print "Warning: unfinished program object"
            p.program = Sequence(p.program)
        
        showImage(p.output)
        print p.program

        result = synthesizeProgram(p.program)
        print result[0]
        print result[1]
        print parseSketchOutput(result[1])
        assert False



def synthesizeFromSequence((parse,whereToPickle)):
    print parse
#    showImage(fastRender(parse))
    startTime = time.time()
    result = synthesizeProgram(parse)
    if result == None: print "Failure to synthesize."
    else:
        result = SynthesisResult(source = result[1],
                                 cost = result[0],
                                 parse = parse,
                                 time = time.time() - startTime)
        pickle.dump(result,open(whereToPickle,'wb'))
    
def synthesizeGroundTruthPrograms(arguments):
    Pool(arguments.cores).map(synthesizeFromSequence,
                             [(groundTruthSequence[k], 'synthesisResults/%s-synthesizerOutput.p'%(k.replace('/','-')))
                              for k in groundTruthSequence ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Synthesis of high-level code from low-level parses')
    parser.add_argument('-d', '--directory', default = None)
    parser.add_argument('-m', '--cores', default = 1, type = int)
    parser.add_argument('--view', default = None, type = str)

    arguments = parser.parse_args()

    if arguments.view:
        viewSynthesisResults(arguments.view)
    elif arguments.directory != None:
        loadParses(arguments.directory)
    else:
        synthesizeGroundTruthPrograms(arguments)
        
