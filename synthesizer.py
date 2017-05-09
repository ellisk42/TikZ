from fastRender import fastRender
from sketch import synthesizeProgram,parseSketchOutput
from language import *
from utilities import showImage,loadImage
from recognitionModel import Particle
from groundTruthParses import groundTruthSequence

import re
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
        latex.append('''
        \\begin{tabular}{ll}
\\includegraphics[width = 5cm]{../TikZ/drawings/expert-%d.png}&
        \\begin{minipage}{10cm}
        \\begin{verbatim}
%s
        \\end{verbatim}
\\end{minipage}
\\end{tabular}        
        '''%(expertIndex, parseSketchOutput(result.source)))
        print

    if arguments.latex:
        latex = '%s'%("\\\\\n".join(latex))
        with open('../TikZpaper/synthesizerOutputs.tex','w') as handle:
            handle.write(latex)
        print "Wrote output to ../TikZpaper/synthesizerOutputs.tex"

        
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
    parser.add_argument('--latex', default = False, action = 'store_true')

    arguments = parser.parse_args()

    if arguments.view:
        viewSynthesisResults(arguments)
    elif arguments.directory != None:
        loadParses(arguments.directory)
    else:
        synthesizeGroundTruthPrograms(arguments)
        
