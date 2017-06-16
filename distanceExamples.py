from language import *
from utilities import *


from random import choice,random
import numpy as np
from time import time

def makeDistanceExamples(targets,programs, continuous = False, reportTime = False):
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
            if not continuous:
                extraCurrent.append(0.0)
                extraTarget.append(len(program) - j)
            else:
                (d1,d2) = smoothDistance(program, Sequence(program.lines[:j]))
                extraCurrent.append(d1)
                extraTarget.append(d2)

        targetShapes = set(map(str,program.lines))

        for _ in range(10):
            exampleTargets.append(target)
            prefixSize = choice(range(len(program) + 1))
            stuff = program.lines
            if random() < 0.4: stuff = np.random.permutation(stuff).tolist()
            mutant = Sequence(stuff[:prefixSize])
            for _ in range(choice(range(max(prefixSize,1)))):
                mutant = mutant.mutate(canRemove = False)
            exampleImages.append(mutant.draw())
            if not continuous:
                mutantShapes = set(map(str,mutant.lines))
                extraTarget.append(len(targetShapes - mutantShapes))
                extraCurrent.append(len(mutantShapes - targetShapes))
            else:
                (d1,d2) = smoothDistance(program, mutant)
                extraCurrent.append(d1)
                extraTarget.append(d2)

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

def smoothDistance(p, q, tolerance = 0.5):

    def d(a,b):
        if not isinstance(a,b.__class__): return None

        if isinstance(a,Line):
            if a.solid != b.solid or a.arrow != b.arrow: return None
            ds = sum([ (x - y).magnitude() for x,y in zip(a.points,b.points) ])
            if ds > tolerance: return None
            return ds

        if isinstance(a,Label):
            if a.c != b.c: return None
            z = (a.p - b.p).magnitude()
            if z > tolerance: return None
            return z

        if isinstance(a,Rectangle):
            z = (a.p1 - b.p1).magnitude() + (a.p2 - b.p2).magnitude()
            if z > tolerance: return None
            return z

        if isinstance(a,Circle):
            z = (a.center - b.center).magnitude() + abs(a.radius - b.radius)
            if z > tolerance: return None
            return z

    # adjacency matrix
    adjacency = ([ [ d(a,b) for b in q.lines ]
                   for a in p.lines ])
    print "adjacency matrix:"
    print adjacency

    def minimumCostAlignment(pIndex, availableQ):
        if pIndex == len(p):
            return (len(availableQ),0.0)
        
        # different things that pIndex might be aligned to
        candidates = [ (j,0,adjacency[pIndex][j]) for j in availableQ
                       if adjacency[pIndex][j] != None ]
        onlyOnePossibility = False
        if len(candidates) == 1:
            qIndex = candidates[0][0]
            if len([ adjacency[j][qIndex] for j in range(len(p)) if adjacency[j][qIndex] != None ]) == 1:
                onlyOnePossibility = True
        if not onlyOnePossibility:
            # match with nothing in Q, 1 not matched, 0.0 continuous alignment cost
            candidates += [(None,1,0.0)]

        bestCost = None
        for (qUsed,mismatchCount,cost) in candidates:
            (recursiveMismatch,recursiveCost) = minimumCostAlignment(pIndex + 1, [j for j in availableQ if j != qUsed ])
            recursiveMismatch += mismatchCount
            recursiveCost += cost
            newCost = (recursiveMismatch,recursiveCost)
            if bestCost == None or newCost < bestCost:
                bestCost = newCost

        return bestCost

    startTime = time()
    mc = minimumCostAlignment(0,range(len(q)))
    print "Calculated minimum cost alignment in %f sec."%(time() - startTime)
    print "mc = ",mc
    return mc
