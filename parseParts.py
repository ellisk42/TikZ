import matplotlib.pyplot as plot
from utilities import *
from distanceMetrics import HausdorffDist,blurredDistance
from GA import GeneticAlgorithm
from render import render,animateMatrices
from language import *
from random import random,choice,seed
from time import time
import numpy as np
import cv2

t = str(time())
seed(t)
print "seed:",t

if False:
    testProgram = '''
    \\node(b)[draw,circle,inner sep=0pt,minimum size = 2cm,ultra thick] at (3,5) {};
    \\node(a)[draw,circle,inner sep=0pt,minimum size = 2cm,ultra thick] at (9,5) {};
    \\draw[ultra thick] (4,5) -- (6,5);
    '''
    plot.imshow(render([testProgram],showImage = False,yieldsPixels = True, resolution = 256)[0],cmap = 'gray')
    plot.show()
    assert False

class InverseRender(GeneticAlgorithm):
    def __init__(self):
        # self.originalProgram = Sequence([Circle(AbsolutePoint(3,3),1),
        #                                  Circle(AbsolutePoint(8,3),1),
        #                                  Circle(AbsolutePoint(3,6),1),
        #                                  Circle(AbsolutePoint(8,6),1),
        #                                  Line([AbsolutePoint(4,3),AbsolutePoint(7,3)], True)])
        # self.target = render([str(self.originalProgram)], showImage = True, yieldsPixels = True)[0]
        self.target = loadImage("challenge.png")
        showImage(self.target)
        showImage(cv2.GaussianBlur(self.target,(61,61),sigmaX = 15))
        self.targetPoints = np.column_stack(np.where(self.target > 0.5))
        self.history = {}

        #print "Program that we are trying to match:",self.originalProgram

        self.cumulativeRenderTime = 0.0

    def randomIndividual(self):
        return Sequence.sample()

    def mutate(self,x):
        return x.mutate()

    def fastPixels(self, programs):
        programs = [p.TikZ() for p in programs]
        toRender = list(set([p for p in programs if not (p in self.history) ]))
        if toRender != []:
            renders = render(toRender, yieldsPixels = True)
            for r,p in zip(renders, toRender):
                self.history[p] = 1.0 - r
        return [self.history[p] for p in programs ]

    def mapFitness(self, programs):
        print "About to get the rendered pixels"

        renderStart = time()
        pixels = self.fastPixels(programs)
        self.cumulativeRenderTime += (time() - renderStart)
        print "Got pixels, calculating distance"

        def f(x): # fitness
            if True:
                # points = np.column_stack(np.where(x > 0.5))
                # if len(points) > 0:
                #     return -HausdorffDist(points, self.targetPoints)
                # else:
                #     return float('-inf')
                return blurredDistance(self.target,x)
            else:
                return -np.sum(np.abs(self.target - x))

        ds = map(f,pixels)
        print "Calculated distance"
        return ds
r = InverseRender()
_,history = r.beam(100, 5, 10)
print "Rendered",len(r.history),"images in",(r.cumulativeRenderTime),"seconds."
animateMatrices([ render([str(x)], yieldsPixels = True)[0] for x in history ],"stochasticHausdorffDist.gif")

