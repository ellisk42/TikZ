from multiprocessing import Pool
import sys
import os
from language import *
from render import render
from PIL import Image
import pickle
from random import choice

CANONICAL = True



def makeSyntheticData(filePrefix, sample, k = 1000, offset = 0):
    """sample should return a program"""
    programs = [sample() for _ in range(k)]
    print "Sampled %d programs."%k
    programPrefixes = [ [ Sequence(p.lines[:j]) for j in range(len(p)+1) ] for p in programs ]
    noisyTargets = [ p.noisyTikZ() for p in programs ]
    distinctPrograms = list(set([ p.TikZ() for prefix in programPrefixes for p in prefix] + noisyTargets))
    pixels = render(distinctPrograms, yieldsPixels = True)
    print "Rendered %d images."%len(distinctPrograms)
    pixels = [ Image.fromarray(ps*255).convert('L') for ps in pixels ]
    pixels = dict(zip(distinctPrograms,pixels))
    
    for j in range(len(programs)):
        pickle.dump(programs[j], open("%s-%d.p"%(filePrefix,j + offset),'wb'))
        pixels[noisyTargets[j]].save("%s-%d-noisy.png"%(filePrefix,j + offset))
        for k in range(len(programs[j])):
            endingPoint = pixels[programPrefixes[j][k+1].TikZ()]
            endingPoint.save("%s-%d-%d.png"%(filePrefix,j + offset,k))

            
def canonicalOrdering(things):
    if things == [] or not CANONICAL: return things
    if isinstance(things[0],Circle):
        # sort the circles so that there are always drawn in a canonical order
        return sorted(things, key = lambda c: (c.center.x.n, c.center.y.n))
    if isinstance(things[0],Line):
        return sorted(things, key = lambda l: (l.points[0].x.n,l.points[0].y.n))
    if isinstance(things[0],Rectangle):
        return sorted(things, key = lambda r: (r.p1.x.n,
                                               r.p1.y.n))

def proposeAttachmentLines(objects):
    attachmentSets = [o.attachmentPoints() for o in objects ]

    for j in range(len(attachmentSets) - 1):
        for k in range(j + 1,len(attachmentSets)):
            for (x1,y1,o1) in attachmentSets[j]:
                for (x2,y2,o2) in attachmentSets[k]:
                    if o1 != o2: continue
                    
                    candidate = None
                    if x2 == x1 and y1 != y2 and o == 'v':
                        candidate = (x1,min(y1,y2),x1,max(y1,y2))
                    elif y2 == y1 and x1 != x2 and o == 'h':
                        candidate = (min(x1,x2),y1,max(x1,x2),y1)
                    if candidate != None:
                        l = Line.absolute(Number(candidate[0]),
                                          Number(candidate[1]),
                                          Number(candidate[2]),
                                          Number(candidate[3]))
                        if all([not o.intersects(l) for o in objects ]):
                            yield candidate

def horizontalOrVerticalLine(attachedLines = []):
    concentration = 0.0
    if attachedLines != [] and random() < float(len(attachedLines))/(concentration + len(attachedLines)):
        (x1,y1,x2,y2) = choice(attachedLines)
        points = [AbsolutePoint(Number(x1),Number(y1)),AbsolutePoint(Number(x2),Number(y2))]
    else:
        x1 = randomCoordinate()
        y1 = randomCoordinate()
        if choice([True,False]):
            # x1 == x2; y1 != y2
            y2 = y1
            while y2 == y1: y2 = randomCoordinate()
            points = [AbsolutePoint(Number(x1),Number(y1)),AbsolutePoint(Number(x1),Number(y2))]
        else:
            # x1 != x2; y1 == y2
            x2 = x1
            while x2 == x1: x2 = randomCoordinate()
            points = [AbsolutePoint(Number(x1),Number(y1)),AbsolutePoint(Number(x2),Number(y1))]
    return Line(list(sorted(points, key = lambda p: (p.x.n,p.y.n))),
                solid = random() > 0.5,
                arrow = random() > 0.5)

def multipleObjects(rectangles = 0,lines = 0,circles = 0):
    def sampler():
        while True:
            cs = canonicalOrdering([ Circle.sample() for _ in range(circles) ])
            rs = canonicalOrdering([ Rectangle.sample() for _ in range(rectangles) ])
            attachedLines = list(proposeAttachmentLines(cs + rs))
            ls = canonicalOrdering([ horizontalOrVerticalLine(attachedLines) for _ in range(lines) ])
            program = cs + rs + ls
            failure = False
            for p in program:
                if failure: break

                for q in program:
                    if p != q and p.intersects(q):
                        failure = True
                        break
            if not failure:
                return Sequence(program)
    return sampler

def randomScene(maximumNumberOfObjects):
    def sampler():
        n = choice(range(maximumNumberOfObjects)) + 1

        shapeIdentities = [choice(range(3)) for _ in range(n) ]
        return multipleObjects(rectangles = len([x for x in shapeIdentities if x == 0 ]),
                               lines = len([x for x in shapeIdentities if x == 1 ]),
                               circles = len([x for x in shapeIdentities if x == 2 ]))()
    return sampler

def handleGeneration(arguments):
    generators = {"individualCircle": multipleObjects(circles = 1),
                  "doubleCircleLine": multipleObjects(circles = 2,lines = 1),
                  "tripleLine": multipleObjects(lines = 3),
                  "doubleCircle": multipleObjects(circles = 2),
                  "randomScene": randomScene(5),
                  "tripleCircle": multipleObjects(circles = 3),
                  "individualRectangle": multipleObjects(rectangles = 1)}
    (n,startingPoint,k) = arguments
    makeSyntheticData("syntheticTrainingData/"+n, generators[n], k = k, offset = startingPoint)
    print "Generated %d training sequences."%k
    
if __name__ == '__main__':
    setCoordinateNoise(0.4)
    setRadiusNoise(0.3)
    k = 10000
    os.system('mkdir syntheticTrainingData')
    for n in sys.argv[1:]:
        print n
        startingPoint = 0
        offsetsAndCounts = []
        while startingPoint < k:
            kp = min(k - startingPoint,1000)
            offsetsAndCounts.append((n,startingPoint,kp))
            startingPoint += 1000
        Pool(5).map(handleGeneration, offsetsAndCounts)
