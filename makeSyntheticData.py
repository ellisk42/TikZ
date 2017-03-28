import sys
import os
from language import *
from render import render
from PIL import Image
import pickle
from random import choice

# It's the synthetic data should look clean: we want people to check that things are not overlapping
def lineIntersectsCircle(l,c):
    x2,y2 = l.points[1].x,l.points[1].y
    x1,y1 = l.points[0].x,l.points[0].y

    if x1 == x2: # vertical line
        y1,y2 = min(y1,y2),max(y1,y2)
        x = x1
        return x == c.center.x and c.center.y > y2 + c.radius and c.center.y < y1 - c.radius
    elif y1 == y2: # horizontal line
        x1,x2 = min(x1,x2),max(x1,x2)
        y = y1
        return y == c.center.y and c.center.x + c.radius < x2 and c.center.x - c.radius > x1
    else:
        raise Exception('arbitrary lines not yet supported')

def lineIntersectsLine(p,q):
    def ccw(A,B,C):
        return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)
    overlapping = ccw(p.points[0],q.points[0],q.points[1]) != ccw(p.points[1],q.points[0],q.points[1]) and ccw(p.points[0],p.points[1],q.points[0]) != ccw(p.points[0],p.points[1],q.points[1])
    touching = p.points[0] == q.points[0] or p.points[1] == q.points[0] or p.points[0] == q.points[1] or p.points[1] == q.points[1]
    return overlapping or touching


def makeSyntheticData(filePrefix, sample, k = 1000):
    """sample should return a program"""
    programs = [sample() for _ in range(k)]
    print "Sampled %d programs."%k
    programPrefixes = [ [ Sequence(p.lines[:j]) for j in range(len(p)+1) ] for p in programs ]
    distinctPrograms = list(set([ str(p) for prefix in programPrefixes for p in prefix]))
    pixels = render(distinctPrograms, yieldsPixels = True)
    print "Rendered %d images."%len(distinctPrograms)
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
        cs = getCircles().lines
        ls = []
        while len(ls) < k:
            l = horizontalOrVerticalLine()
            # check to intersect any of the circles
            failure = False
            for c in cs:
                if lineIntersectsCircle(l,c):
                    failure = True
                    break
            for c in ls:
                if lineIntersectsLine(c,l):
                    failure = True
                    break
                
            if not failure: ls.append(l)
        ls = sorted(ls, key = lambda l: (l.points[0].x,l.points[0].y))
        return Sequence(cs + ls)
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
    #}challenge program
    challenge = '''
    \\node(b)[draw,circle,inner sep=0pt,minimum size = 2cm,ultra thick] at (5,2) {};
    \\node(a)[draw,circle,inner sep=0pt,minimum size = 2cm,ultra thick] at (5,6) {};
    \\node(a)[draw,circle,inner sep=0pt,minimum size = 2cm,ultra thick] at (2,6) {};
    \\node(a)[draw,circle,inner sep=0pt,minimum size = 2cm,ultra thick] at (2,2) {};
    \\draw[ultra thick] (5,3) -- (5,5);
    \\draw[ultra thick] (2,3) -- (2,5);
    '''
    Image.fromarray(255*render([challenge],showImage = False,yieldsPixels = True, resolution = 256)[0]).convert('L').save('challenge.png')
    
    generators = {"randomObjects": randomObjects(4),
                  "individualCircle": multipleCircles(1),
                  "doubleCircleLine": circlesAndLine(2,1),
                  "doubleLine": circlesAndLine(0,2),
                  "doubleCircle": multipleCircles(2),
                  "tripleCircle": multipleCircles(3)}
    n = sys.argv[1]
    makeSyntheticData("syntheticTrainingData/"+n, generators[n])

