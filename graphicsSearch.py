from DSL import *
from neuralSearch import *

from dispatch import dispatch

import random
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.optim as optimization
import torch.cuda as cuda

GPU = cuda.is_available()







class GraphicsSearchPolicy(SearchPolicy):
    def __init__(self):
        LEXICON = ["START","END",
                   "circle",
                   "rectangle",
                   "line","arrow = True","arrow = False","solid = True","solid = False",
                   "for",
                   "reflect","x","y",
                   "i","j","None"] + map(str,range(-5,20))
        super(GraphicsSearchPolicy,self).__init__(LEXICON)

        self.circleEncoder = nn.Linear(2, self.H)
        self.interactionEncoder1 = nn.Linear(self.H*2,self.H)
        self.interactionEncoder2 = nn.Linear(self.H,self.H)

    def encodeProblem(self, s):
        encodings = [ self.circleEncoder(Variable(t.cuda() if GPU else t)).clamp(min = 0) \
                      for c in s\
                      for t in [torch.from_numpy(np.array([c.center.x, c.center.y])).float()] ]
        interactions = [self.interactionEncoder1(torch.cat([x,y],dim = 0)).clamp(min = 0)
                        for i,x in enumerate(encodings)
                        for j,y in enumerate(encodings)]
        interactions = [self.interactionEncoder2(interaction).clamp(min = 0)
                        for interaction in interactions ]
        return sum(interactions)

    def candidateEnvironments(self, program):
        return list(e for e in candidateEnvironments(program))

    def applyChange(self, program, environment, line):
        if isinstance(program, Loop):
            return Loop(body = self.applyChange(program.body, environment, line),
                        v = program.v,
                        bound = program.bound)
        if isinstance(program, Reflection):
            return Reflection(body = self.applyChange(program.body, environment, line),
                              axis = program.axis,
                              coordinate = program.coordinate)

        assert isinstance(program, Block)
        
        if environment == []:
            return Block([self.parseLine(line)] + program.items)
        # Figure out what it is that's being indexed into
        for j,l in enumerate(program.items):
            s = serializeLine(l)
            if s == environment[:len(s)]:
                lp = self.applyChange(l, environment[len(s):], line)
                newItems = list(program.items)
                newItems[j] = lp
                return Block(newItems)
        raise Exception('Environment indexes nonexistent context')
    
    def Oracle(self, program): return list(Oracle(program))

    def evaluate(self, program):
        try: return program.evaluate(Environment([]))
        except EvaluationError: return None
    def solvesTask(self, goal, program):
        return goal == self.evaluate(program)
    def residual(self, goal, current):
        #assert len(current - goal) == 0
        return goal - current
    def value(self, goal, program):
        try:
            output = self.evaluate(program)
        except EvaluationError: return -1.0
        if output == None: return -1.0
        if len(output - goal) > 0: return 0.0
        else: return 1.0/program.cost()

    def parseLine(self, l):
        def get(l):
            n = l[0]
            del l[0]
            return n
        def finish(l):
            if l != []: raise Exception('Extra symbols in line')
        def parseLinear(l):
            b = int(get(l))
            x = get(l)
            m = int(get(l))
            if x == 'None': x = None
            return LinearExpression(m,x,b)
        k = get(l)
        if k == 'circle':
            x = parseLinear(l)
            y = parseLinear(l)
            finish(l)
            return Primitive(k,x,y)
        if k == 'for':
            v = get(l)
            b = parseLinear(l)
            finish(l)
            return Loop(v = v, bound = b, body = Block([]))
        if k == 'reflect':
            a = get(l)
            c = int(get(l))
            finish(l)
            return Reflection(body = Block([]), axis = a, coordinate = c)
        raise Exception('parsing line '+k)
            

    
        
        
@dispatch(Block)
def Oracle(b):
    for j,x in enumerate(b.items):
        serialized = serializeLine(x)
        yield Block(b.items[:j]), [], serialized
        for program, environment, line in Oracle(x):
            yield Block(b.items[:j] + [program]), serialized + environment, line
@dispatch(Primitive)
def Oracle(p):
    return
    yield
@dispatch(Loop)
def Oracle(l):
    for program, environment, line in Oracle(l.body):
        yield Loop(v = l.v, bound = l.bound, body = program), environment, line
@dispatch(Reflection)
def Oracle(l):
    for program, environment, line in Oracle(l.body):
        yield Reflection(axis = l.axis, coordinate = l.coordinate, body = program), environment, line
    


@dispatch(Loop)
def serializeLine(l):
    return ["for",l.v] + serializeLine(l.bound)
@dispatch(Reflection)
def serializeLine(r):
    return ["reflect",r.axis,str(r.coordinate)]
@dispatch(LinearExpression)
def serializeLine(e):
    return [str(e.b),str(e.x),str(e.m)]
@dispatch(Primitive)
def serializeLine(p):
    s = [p.k]
    for a in p.arguments[:4]:
        s += serializeLine(a)
    if p.k == 'line':
        s += ["arrow = True" if "True" in p.arguments[4] else "arrow = False" ]
        s += ["solid = True" if "True" in p.arguments[5] else "solid = False"]
    return s

@dispatch(Circle)
def serializeObservation(c):
    return ["circle",str(c.center.x),str(c.center.y)]
@dispatch(Rectangle)
def serializeObservation(c):
    return ["rectangle",str(c.p1.x),str(c.p1.y),str(c.p2.x),str(c.p2.y)]

@dispatch(Block)
def candidateEnvironments(b):
    yield []
    for x in b.items:
        for e in candidateEnvironments(x):
            yield e
@dispatch(Primitive)
def candidateEnvironments(_):
    return
    yield 
@dispatch(Loop)
def candidateEnvironments(l):
    this = serializeLine(l)
    for e in candidateEnvironments(l.body):
        yield this + e
@dispatch(Reflection)
def candidateEnvironments(r):
    this = serializeLine(l)
    for e in candidateEnvironments(l.body):
        yield this + e

def simpleSceneSample():
    def isolatedCircle():
        x = random.choice(range(1,16))
        y = random.choice(range(1,16))
        return Primitive('circle', LinearExpression(0,None,x), LinearExpression(0,None,y))

    MINIMUMATOMS = 1
    MAXIMUMATOMS = 1
    primitives = [isolatedCircle() for _ in range(random.choice(range(MINIMUMATOMS,MAXIMUMATOMS+1))) ]
    loopIterations = random.choice([4])

    while True:
        bx = random.choice(range(1,16))
        mx = random.choice(range(-5,6))
        if all([x > 0 and x < 16 for j in range(loopIterations) for x in [mx*j + bx] ]): break
    while True:
        by = random.choice(range(1,16))
        my = random.choice(range(-5,6))
        if my == 0 and mx == 0: continue
        
        if all([y > 0 and y < 16 for j in range(loopIterations) for y in [my*j + by] ]): break



    l = Loop(v = 'j',bound = LinearExpression(0,None,loopIterations),
             body = Block([Primitive('circle',
                                     LinearExpression(mx,'j' if mx else None,bx),
                                     LinearExpression(my,'j' if my else None,by))]))
    return Block([l] + primitives)
    
    
if __name__ == "__main__":
    p = GraphicsSearchPolicy()
    if GPU:
        print "Using the GPU"
        p.cuda()

    o = optimization.Adam(p.parameters(), lr = 0.001)
    
    step = 0
    losses = []
    while True:
        step += 1

        
        program = simpleSceneSample()
        scene = set(program.convertToSequence().lines)

        examples = p.makeOracleExamples(program, scene)
        
        for example in examples:
            o.zero_grad()
            loss = p.loss(example)
            loss.backward()
            o.step()
            losses.append(loss.data[0])

        if step%100 == 0:
            print "LOSS:", step,'\t',sum(losses)/len(losses)
            losses = []
        if step%5000 == 0:
            torch.save(p.state_dict(),'checkpoints/neuralSearch.p')
            print scene
            print program.pretty()
            print p.Oracle(program)
            p0 = Block([])
            p.beamSearchGraph(scene, p0, 30, 3)
            continue
            for _ in range(5):
                p0 = p.sampleOneStep(scene, p0)
                print p0
                try:
                    denotation = p.evaluate(p0)
                except EvaluationError:
                    print "Error evaluating that program"
                    break
                if len(scene - denotation) == 0:
                    print "Nothing left to explain."
                    break
                
