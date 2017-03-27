import sys
import os
from language import *
from render import render
from PIL import Image
import pickle
from random import choice

def makeSyntheticData(filePrefix, sample, k = 1000):
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

def horizontalOrVerticalLine():
    x1 = randomCoordinate()
    y1 = randomCoordinate()
    if choice([True,False]):
        y2 = y1
        while y2 == y1: y2 = randomCoordinate()
        points = [AbsolutePoint(x1,y1),AbsolutePoint(x1,y2)]
    else:
        x2 = x1
        while x2 == x1: x2 = randomCoordinate()
        points = [AbsolutePoint(x1,y1),AbsolutePoint(x2,y1)]
    return Line(list(sorted(points, key = lambda p: (p.x,p.y))))

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

        return Sequence(p.lines + [horizontalOrVerticalLine() for _ in range(k) ])
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


if __name__ == '__main__':
    generators = {"randomObjects": randomObjects(4),
                  "individualCircle": multipleCircles(1),
                  "doubleCircleLine": circlesAndLine(2,1),
                  "doubleCircle": multipleCircles(2),
                  "tripleCircle": multipleCircles(3)}
    n = sys.argv[1]
    makeSyntheticData("syntheticTrainingData/"+n, generators[n])

