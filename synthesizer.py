from sketch import synthesizeProgram
from language import *
from utilities import showImage
from recognitionModel import Particle

import os
import argparse
import pickle

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

        synthesizeProgram(p.program)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Synthesis of high-level code from low-level parses')
    parser.add_argument('directory')

    arguments = parser.parse_args()
    loadParses(arguments.directory)
    
