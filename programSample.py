from utilities import *
from language import *
from DSL import *
from CRP import *

from dispatch import dispatch

import random

class SampleEnvironment():
    def __init__(self, freeVariables, rx = None, ry = None, rl = None):
        self.freeVariables = freeVariables
        assert (rx == None) == (ry == None)
        self.rx = rx if rx != None else ChineseRestaurant(2.0, lambda : random.choice([-4,-3,-2,-1,1,2,3,4]))
        self.ry = ry if ry != None else ChineseRestaurant(2.0, lambda : random.choice([-4,-3,-2,-1,1,2,3,4]))
        self.rl = rl or ChineseRestaurant(2.0, lambda : random.choice([1,2]))

    def introduceFreeVariable(self,v):
        return SampleEnvironment([v] + self.freeVariables, self.rx, self.ry, self.rl)

    def deepCopy(self):
        return SampleEnvironment(self.freeVariables,
                                 self.rx.copy(),
                                 self.ry.copy(),
                                 self.rl.copy())

def sampleLinearExpression(coefficientRestaurant, freeVariables):
    if freeVariables == [] or random.random() < 0.3: return LinearExpression(0,None,random.choice(range(1,15)))

    return LinearExpression(coefficientRestaurant.sampleNew(),
                            random.choice(freeVariables),
                            random.choice(range(1,15)))
def samplePoint(e):
    return (sampleLinearExpression(e.rx,e.freeVariables),sampleLinearExpression(e.ry,e.freeVariables))

def sampleCircle(e):
    x,y = samplePoint(e)
    return Primitive('circle',x,y)
def sampleRectangle(e):
    while True:
        x1,y1 = samplePoint(e)
        x2,y2 = samplePoint(e)
        if x1 != x2 and y1 != y2:
            return Primitive('rectangle',x1,y1,x2,y2)
def sampleLine(e):
    while True:
        x1,y1 = samplePoint(e)
        x2,y2 = samplePoint(e)
        if x1 != x2 or y1 != y2:
            return Primitive('line',x1,y1,x2,y2,
                             random.random() > 0.5,
                             random.random() > 0.5)

def samplePrimitive(e):
    q = random.random()
    if q < 0.33: return sampleCircle(e)
    elif q < 0.66: return sampleLine(e)
    else: return sampleRectangle(e)

def sampleLoop(e):
    b = sampleLinearExpression(e.rl, e.freeVariables)
    if b.m == 0:
        b.b = random.choice(range(3,5))
    else:
        b.b = random.choice(range(1,4))
    return Loop(chr(ord('i') + len(e.freeVariables)),
                b,
                Block([]),
                boundary = Block([]))
def sampleReflection(e):
    return Reflection(random.choice(['x','y']),
                      random.choice(range(1,20)),
                      Block([]))

@dispatch(SampleEnvironment,Block)
def mutateProgram(e,p):
    n = random.choice(range(len(p.items)+1))
    if n < len(p.items) and not isinstance(p.items[n],Primitive):
        new = mutateProgram(e,p.items[n])
        return Block(p.items[:n] + [new] + p.items[n+1:])
    else:
        u = random.random()
        if len(e.freeVariables) <= 1 and u < 0.3: new = sampleLoop(e)
        elif u < 0.75: new = samplePrimitive(e)
        else: new = sampleReflection(e)
        return Block(p.items + [new])

@dispatch(SampleEnvironment,Primitive)
def mutateProgram(e,p): return p
@dispatch(SampleEnvironment,Loop)
def mutateProgram(e,p):
    ep = e.introduceFreeVariable(p.v)
    if random.random() < 0.5: return Loop(p.v,p.bound,mutateProgram(ep,p.body),boundary = p.boundary)
    else: return Loop(p.v,p.bound,p.body,boundary = mutateProgram(ep,p.boundary))
@dispatch(SampleEnvironment,Reflection)
def mutateProgram(e,p):
    return Reflection(p.axis,p.coordinate,mutateProgram(e,p.body))


def randomPrograms(mutations = 200):
    ps = []
    p = Block([])
    e = SampleEnvironment([])
    for _ in range(mutations):
        oldEnvironment = e.deepCopy()
        oldProgram = p
        p = mutateProgram(e,p)
        output = p.convertToSequence()
        if output.extentInWindow() and not output.hasCollisions():
            ps.append(p.removeDeadCode().optimizeUsingRewrites()[1].canonical())
        else:
            p = oldProgram
            e = oldEnvironment
    return ps

def sampleManyPrograms(seed):
    random.seed(seed)
    timeout = 2
    from time import time
    start = time()
    ps = []
    while time() - start < timeout*60*60:
        ps += randomPrograms()
    print "Returning",len(ps),"random programs"
    return ps


if __name__ == "__main__":
    from multiprocessing import Pool
    C = 30
    T = 4
    results = Pool(C).map(sampleManyPrograms, range(C))
    ps = []
    for r in results: ps += r
    

    import cPickle as pickle
    with open('randomlyGeneratedPrograms.p','wb') as handle:
        pickle.dump(ps, handle, protocol = -1)
    print "Collected",len(ps),"random programs altogether"
