import matplotlib.pyplot as plot
from utilities import *
from distanceMetrics import HausdorffDist,blurredDistance,asymmetricBlurredDistance
from GA import GeneticAlgorithm
from render import render,animateMatrices
from language import *
from random import random,choice,seed
from fastRender import fastRender,loadPrecomputedRenderings
from time import time
import numpy as np
import cv2

t = str(time())
seed(t)
print "seed:",t



class InverseRender(GeneticAlgorithm):
    def __init__(self):
        # self.originalProgram = Sequence([Circle(AbsolutePoint(3,3),1),
        #                                  Circle(AbsolutePoint(8,3),1),
        #                                  Circle(AbsolutePoint(3,6),1),
        #                                  Circle(AbsolutePoint(8,6),1),
        #                                  Line([AbsolutePoint(4,3),AbsolutePoint(7,3)], True)])
        # self.target = render([str(self.originalProgram)], showImage = True, yieldsPixels = True)[0]
        self.target = loadImage("challenge.png")
        self.history = {}

        #print "Program that we are trying to match:",self.originalProgram

        self.cumulativeRenderTime = 0.0
        loadPrecomputedRenderings()

    def randomIndividual(self):
        return Sequence.sample()

    def mutate(self,x):
        return x.mutate()

    def fastPixels(self, programs):
        return [fastRender(p) for p in programs ]
    def mapFitness(self, programs):

        renderStart = time()
        pixels = self.fastPixels(programs)
        self.cumulativeRenderTime += (time() - renderStart)

        def f(x): # fitness
            return -asymmetricBlurredDistance(self.target,x)

        ds = map(f,pixels)
        return ds
r = InverseRender()
_,history = r.beam(100000, 5, 10)
print "Rendered",len(r.history),"images in",(r.cumulativeRenderTime),"seconds."
animateMatrices([ fastRender(x) for x in history ],"stochasticGeneticAlgorithm.gif")

