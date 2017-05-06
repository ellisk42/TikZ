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
    for f in os.listdir(directory):
        if f.endswith('.p'):
            particles.append(pickle.load(open(directory + '/' + f,'rb')))
            print " [+] Loaded %s/%s"%(directory,f)

            

    particles.sort(key = lambda p: p.logLikelihood,reverse = True)

    for p in particles:
        if not p.finished():
            print "Warning: unfinished program object"
            p.program = Sequence(p.program)
#        if len(p.program.lines) != 6: continue
        
        showImage(p.output)
        print p.program

        synthesizeProgram(p.program)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Synthesis of high-level code from low-level parses')
    parser.add_argument('directory')

    arguments = parser.parse_args()
    loadParses(arguments.directory)
    
