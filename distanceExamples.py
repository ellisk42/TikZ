from language import *
from fastRender import fastRender
from utilities import *


from random import choice
from time import time

def makeDistanceExamples(targets,programs):
    startTime = time()
    
    # Each example will be a tuple of (image, extraLinesAndTarget, extraLinesInCurrent)
    exampleTargets = []
    exampleImages = []
    extraTarget = []
    extraCurrent = []

    for target, program in zip(targets, programs):
        program = program.item()
        # on policy examples
        for j in range(len(program)+1):
            exampleTargets.append(target)
            exampleImages.append(fastRender(Sequence(program.lines[:j])))
            extraCurrent.append(0.0)
            extraTarget.append(len(program) - j)

        targetShapes = set(map(str,program.lines))

        for _ in range(10):
            exampleTargets.append(target)
            prefixSize = choice(range(len(program) + 1))
            mutant = Sequence(program.lines[:prefixSize])
            for _ in range(choice(range(max(prefixSize,1)))):
                mutant = mutant.mutate()
            exampleImages.append(fastRender(mutant))
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

#    print "Generated examples from %d programs in %f seconds"%(len(programs),time() - startTime)
    return exampleTargets, np.array(exampleImages), t
