from language import *
from utilities import *


from random import choice,random
import numpy as np
from time import time

def makeDistanceExamples(targets,programs, reportTime = False):
    startTime = time()
    
    # Each example will be a tuple of (image, extraLinesAndTarget, extraLinesInCurrent)
    exampleTargets = []
    exampleImages = []
    extraTarget = []
    extraCurrent = []

    for target, program in zip(targets, programs):
        if not isinstance(program, Sequence):
            print "Fatal error: one of the entries of programs was not a sequence"
            print "programs = ",programs
            print "program = ",program
            assert False
        # on policy examples
        for j in range(len(program)+1):
            exampleTargets.append(target)
            exampleImages.append(Sequence(program.lines[:j]).draw())
            extraCurrent.append(0.0)
            extraTarget.append(len(program) - j)

        targetShapes = set(map(str,program.lines))

        for _ in range(10):
            exampleTargets.append(target)
            prefixSize = choice(range(len(program) + 1))
            stuff = program.lines
            if random() < 0.4: stuff = np.random.permutation(stuff).tolist()
            mutant = Sequence(stuff[:prefixSize])
            for _ in range(choice(range(max(prefixSize,1)))):
                mutant = mutant.mutate()
            exampleImages.append(mutant.draw())
            mutantShapes = set(map(str,mutant.lines))
            extraTarget.append(len(targetShapes - mutantShapes))
            extraCurrent.append(len(mutantShapes - targetShapes))
    exampleTargets = augmentData(np.array(exampleTargets))
    if False:
        for j in range(len(extraCurrent)):
            print extraTarget[j]
            print extraCurrent[j]
            showImage(np.concatenate((exampleTargets[j],exampleImages[j]),
                                     axis = 0))
            print
            print 
    t = np.stack([np.array(extraTarget), np.array(extraCurrent)],axis = 1)

    if reportTime: print "Generated examples from %d programs in %f seconds"%(len(programs),time() - startTime)
    return exampleTargets, np.array(exampleImages), t
