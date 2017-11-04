from DSL import *

from dispatch import dispatch

import random
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.optim as optimization

class DataEncoder(nn.Module):
    """
    A network that takes something like circle(9,2) and produces an embedding
    """
    def __init__(self, numberOfInputs, embeddingSize):
        super(DataEncoder,self).__init__()

        self.linear1 = nn.Linear(numberOfInputs, embeddingSize)

    def forward(self, x):
        return self.linear1(x).clamp(min = 0)

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
        


class SearchPolicy(nn.Module):
    def __init__(self):
        super(SearchPolicy,self).__init__()
        
        observationEmbeddingSize = 20
        
        self.circleEncoder = DataEncoder(2, observationEmbeddingSize)
        self.rectangleEncoder = DataEncoder(4, observationEmbeddingSize)

        self.x1decoder = ExpressionDecoder(range(-4,5), range(0,17), observationEmbeddingSize)
        self.y1decoder = ExpressionDecoder(range(-4,5), range(0,17), observationEmbeddingSize)

    def encodeObservation(self, o):
        if isinstance(o,Circle):
            return self.circleEncoder(Variable(torch.from_numpy(np.array([o.center.x, o.center.y]).astype(np.float32))))
        if isinstance(o,Rectangle):
            return self.rectangleEncoder(Variable(torch.from_numpy(np.array([o.p1.x, o.p1.y,
                                                                             o.p2.x, o.p2.y]))))
        assert False

    def encodeScene(self, scene):
        return sum(self.encodeObservation(o) for o in scene )

    def crossEntropyLoss(self, scene, target):
        embedding = self.encodeScene(scene)
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

    
        
        
