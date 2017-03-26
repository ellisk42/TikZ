import os
from language import *
from render import render
from PIL import Image
import pickle
from random import choice

def makeSyntheticData(sample, filePrefix, k = 1000):
    """sample should return a program"""
    programs = [sample() for _ in range(k)]
    programPrefixes = [ [ Sequence(p.lines[:j]) for j in range(len(p)+1) ] for p in programs ]
    distinctPrograms = list(set([ str(p) for prefix in programPrefixes for p in prefix]))
    pixels = render(distinctPrograms, yieldsPixels = True)
    pixels = [ Image.fromarray(ps*255).convert('L') for ps in pixels ]
    pixels = dict(zip(distinctPrograms,pixels))
    
    for j in range(len(programs)):
        pickle.dump(programs[j], open("%s-%d.p"%(filePrefix,j),'wb'))
        for k in range(len(programs[j])):
            endingPoint = pixels[programPrefixes[j][k+1]]
            endingPoint.save("%s-%d-%d.png"%(filePrefix,j,k))
            
def canonicalOrdering(circles):
    # sort the circles so that there are always drawn in a canonical order
    return sorted(circles, key = lambda c: (c.center.x, c.center.y))

def multipleCircles(n):
    def sampler():
        while True:
            p = [Circle.sample() for _ in range(n) ]
            if all([ a == b or (a.center.x-b.center.x)**2 + (a.center.y-b.center.y)**2 > 4
                    for a in p
                    for b in p ]):
                return Sequence(canonicalOrdering(p))
    return sampler

def circlesAndLine(n,k):
    getCircles = multipleCircles(n)
    def sampler():
        p = getCircles()

        linePoints = [AbsolutePoint.sample(),AbsolutePoint.sample()]
        linePoints.sort(key = lambda p: (p.x,p.y))

        return Sequence(p.lines + [Line(linePoints)])
    return sampler

def randomObjects(n):
    def sample():
        nl = choice([0,1,2])
        nc = n - nl
        p = multipleCircles(nc)()

        lines = [ Line(sorted([AbsolutePoint.sample(),AbsolutePoint.sample()],
                              key = lambda p: (p.x,p.y))) for _ in range(nl) ]
        lines.sort(key = lambda l: (l.points[0].x,l.points[0].y))

        return Sequence(p.lines + lines)
    return sample


makeSyntheticData(randomObjects(4), "syntheticTrainingData/randomObjects", 1000)
makeSyntheticData(circlesAndLine(2,1), "syntheticTrainingData/doubleCircleLine", 1000)
makeSyntheticData(multipleCircles(1), "syntheticTrainingData/individualCircle", 1000)
makeSyntheticData(multipleCircles(2), "syntheticTrainingData/doubleCircle", 1000)
makeSyntheticData(multipleCircles(3), "syntheticTrainingData/tripleCircle", 1000)

