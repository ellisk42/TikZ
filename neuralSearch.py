from DSL import *

from dispatch import dispatch

import random
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.optim as optimization

class LineDecoder(nn.Module):
    def __init__(self, lexicon, embeddingSize = 30):
        super(self.__class__,self).__init__()
        self.lexicon = lexicon
        self.model = nn.LSTM

    def sample():

class DataEncoder(nn.Module):
    """
    A network that takes something like circle(9,2) and produces an embedding
    """
    def __init__(self, numberOfInputs, embeddingSize):
        super(DataEncoder,self).__init__()

        self.linear1 = nn.Linear(numberOfInputs, embeddingSize)

    def forward(self, x):
        return self.linear1(x).clamp(min = 0)

class ExpressionEncoder(nn.Module):
    """
    Converts m*x+b to a vector
    """
    def __init__(self, embeddingSize, H = 20):
        super(ExpressionEncoder,self).__init__()
        self.linear1 = nn.Linear(3, H)
        self.linear2 = nn.Linear(H, embeddingSize)

    def forward(self, x):
        return self.linear2(self.linear1(x).clamp(min = 0)).clamp(min = 0)

    def variableOfExpression(self, e):
        v = 0
        if e.v != None:
            v = ord(e.v) - ord('i') + 1
        return Variable(torch.from_numpy(np.array([e.m, v, e.b]).astype(np.float32)))

class ReflectionEncoder(nn.Module):
    """
    Converts reflect([x|y] = c) to a vector
    """
    def __init__(self, embeddingSize, H = 20):
        super(ReflectionEncoder,self).__init__()
        self.linear1 = nn.Linear(3, H)
        self.linear2 = nn.Linear(H, embeddingSize)

    def forward(self, x):
        return self.linear2(self.linear1(x).clamp(min = 0)).clamp(min = 0)

    def variableOfReflection(self, e):
        return Variable(torch.from_numpy(np.array([int(e.axis == 'x'), e.coordinate]).astype(np.float32)))

   
class LoopEncoder(nn.Module):
    def __init__(self, embeddingSize, H = 20):
        super(self.__class__,self).__init__()
        self.linear1 = nn.Linear(embeddingSize, H)
        self.linear2 = nn.Linear(H, embeddingSize)

    def forward(self, x):
        """
        x: embedding of the loop iteration expression
        """
        return self.linear2(self.linear1(x).clamp(min = 0)).clamp(min = 0)

class ReflectionDecoder(nn.Module):
    """
    Converts and embedding to a distribution over things like reflect([x|y] = c)
    """
    def __init__(self, coordinateRange, embeddingSize, H = 20):
        super(self.__class__,self).__init__()
        self.linear1 = nn.Linear(embeddingSize, H)
        self.axis = nn.Linear(embeddingSize, 2)
        self.coordinate = nn.Linear(embeddingSize, len(coordinateRange))

        self.coordinateRange = coordinateRange
        self.coordinate2target = dict(zip(coordinateRange,range(100)))

    def forward(self, x):
        h = self.linear1(x).clamp(min = 0)
        return F.log_softmax(self.axis(h)), F.log_softmax(self.coordinate(h))

    def crossEntropyLoss(self, x, a,c):
        a_,c_ = self.forward(x)
        return  - (a_.dot(a) + c_.dot(c))

    def targetsOfReflection(self, r):
        at = torch.zeros(2)
        if r.axis == 'x': at[0] = 1.0
        else: at[1] = 1.0
        ct = torch.zeros(len(self.coordinateRange))
        ct[self.coordinate2target[r.coordinate]] = 1.0
        return Variable(at),Variable(ct)

    def beam(self, e):
        a,c = self.forward(e)
        return sorted(( (a.data[slopeIndex] + c.data[interceptIndex],
                         Reflection(axis, coordinate)) \
                         for axisIndex, axis in enumerate(['x','y']) \
                         for coordinateIndex, coordinate in enumerate(self.coordinateRange)), \
                       reverse = True)


class ExpressionDecoder(nn.Module):
    """
    A network that predicts a distribution over linear expressions
    """
    def __init__(self, slopeRange, interceptRange, inputDimension):
        super(ExpressionDecoder, self).__init__()

        self.slopeRange = slopeRange
        self.interceptRange = interceptRange

        self.slope2target = dict(zip(self.slopeRange, range(100)))
        self.intercept2target = dict(zip(self.interceptRange, range(100)))

        self.slopeDecoder = nn.Linear(inputDimension, len(slopeRange))
        self.interceptDecoder = nn.Linear(inputDimension, len(interceptRange))

    def forward(self, x):
        return F.log_softmax(self.slopeDecoder(x)),\
            F.log_softmax(self.interceptDecoder(x))

    def crossEntropyLoss(self, x, slopeTarget, interceptTarget):
        m,b = self.forward(x)
        return - (m.dot(slopeTarget) + b.dot(interceptTarget))

    def targetsOfExpression(self, e):
        mt = torch.zeros(len(self.slopeRange))
        mt[self.slope2target[e.m]] = 1.0
        bt = torch.zeros(len(self.interceptRange))
        bt[self.intercept2target[e.b]] = 1.0
        return Variable(mt),Variable(bt)
    

    def beam(self, e):
        m,b = self.forward(e)
        return sorted(( (m.data[slopeIndex] + b.data[interceptIndex],
                        LinearExpression(slope,None if slope == 0 else 'j',intercept)) \
                         for slopeIndex, slope in enumerate(self.slopeRange) \
                         for interceptIndex, intercept in enumerate(self.interceptRange)), \
                       reverse = True)
        
class ContextDecider(nn.Module):
    """Scores how good it is to add a line of code to a given context"""
    def __init__(self, embeddingSize, H = 10):
        super(self.__class__,self).__init__()

        self.linear1 = nn.Linear(embeddingSize*2,H)
        self.linear2 = nn.Linear(H,1)

    def forward(self, problem, context):
        h = self.linear1(torch.cat([problem, context])).clamp(min = 0)
        return self.linear2(h)

class SearchPolicy(nn.Module):
    def __init__(self):
        super(SearchPolicy,self).__init__()
        
        observationEmbeddingSize = 20
        
        self.circleEncoder = DataEncoder(2, observationEmbeddingSize)
        self.rectangleEncoder = DataEncoder(4, observationEmbeddingSize)

        self.loopEncoder = LoopEncoder(observationEmbeddingSize)
        self.reflectionEncoder = ReflectionEncoder(observationEmbeddingSize)

        self.x1decoder = ExpressionDecoder(range(-4,5), range(0,17), observationEmbeddingSize)
        self.y1decoder = ExpressionDecoder(range(-4,5), range(0,17), observationEmbeddingSize)

        self.contextDecider = ContextDecider(observationEmbeddingSize)

    def contextVector(self, context):
        if isinstance(context,list): return sum(self.contextVector(p) for p in context )
        if isinstance(context,Loop): return self.loopEncoder(self)

    def encodeObservation(self, o):
        if isinstance(o,Circle):
            return self.circleEncoder(Variable(torch.from_numpy(np.array([o.center.x, o.center.y]).astype(np.float32))))
        if isinstance(o,Rectangle):
            return self.rectangleEncoder(Variable(torch.from_numpy(np.array([o.p1.x, o.p1.y,
                                                                             o.p2.x, o.p2.y]))))
        assert False

    def encodeScene(self, scene):
        return sum(self.encodeObservation(o) for o in scene )

    def crossEntropyLoss(self, scene, target, targetContext, otherContexts):
        embedding = self.encodeScene(scene)

        self.contextDecider()
        
        if isinstance(target,Primitive) and target.k == 'circle':
            l = self.x1decoder.crossEntropyLoss(embedding,
                                                *self.x1decoder.targetsOfExpression(target.arguments[0]))
            l += self.y1decoder.crossEntropyLoss(embedding,
                                                 *self.y1decoder.targetsOfExpression(target.arguments[1]))
            return l
        assert False

    def beam(self, e):
        x1 = self.x1decoder.beam(e)
        y1 = self.y1decoder.beam(e)

        return sorted( ((xl + yl, Primitive('circle', x, y))\
                       for xl,x in x1 for yl,y in y1), \
                       reverse = True)
        
        
        

if __name__ == "__main__":
    p = SearchPolicy()
    o = optimization.Adam(p.parameters(), lr = 0.001)
    
    for step in range(10000):
        x = random.choice(range(1,4))
        y = random.choice(range(1,4))
        scene = [Circle.absolute(x,y)]
        target = Primitive('circle',LinearExpression(0,None,x),LinearExpression(0,None,y))

        o.zero_grad()
        loss = p.crossEntropyLoss(scene, target)
        loss.backward()
        o.step()

        if step%100 == 0:
            print step,'\t',loss.data[0]
            print p.beam(p.encodeScene(scene))[0][1],scene[0]

    
        
        
