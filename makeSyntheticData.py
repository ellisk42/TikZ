from multiprocessing import Pool
import sys
import os
from language import *
from render import render
from PIL import Image
import pickle
from random import choice,shuffle
from utilities import showImage

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

    # for program,image in zip(distinctPrograms,pixels):
    #     print program
    #     showImage(image)
    #     print 
    
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
    cs = [c for c in things if isinstance(c,Circle) ]
    cs = sorted(cs, key = lambda c: (c.center.x.n, c.center.y.n))
    rs = [c for c in things if isinstance(c,Rectangle) ]
    rs = sorted(rs, key = lambda r: (r.p1.x.n,
                                         r.p1.y.n))
    ls = [c for c in things if isinstance(c,Line) ]
    ls = sorted(ls, key = lambda l: (l.points[0].x.n,l.points[0].y.n))
    return cs + rs + ls


def proposeAttachmentLines(objects):
    attachmentSets = [o.attachmentPoints() for o in objects ]

    # attachments where they both have the same orientation and are nicely aligned
    alignedAttachments = []
    # all the other possible attachments
    arbitraryAttachments = []

    for j in range(len(attachmentSets) - 1):
        for k in range(j + 1,len(attachmentSets)):
            for (x1,y1,o1) in attachmentSets[j]:
                for (x2,y2,o2) in attachmentSets[k]:
                    
                    candidate = None
                    isAligned = True
                    if x2 == x1 and y1 != y2 and o1 == 'v' and o2 == 'v':
                        candidate = (x1,min(y1,y2),x1,max(y1,y2))
                    elif y2 == y1 and x1 != x2 and o1 == 'h' and o2 == 'h':
                        candidate = (min(x1,x2),y1,max(x1,x2),y1)
                    else:
                        candidate = (x1,y1,x2,y2)
                        isAligned = False
                        
                    if candidate != None:
                        l = Line.absolute(Number(candidate[0]),
                                          Number(candidate[1]),
                                          Number(candidate[2]),
                                          Number(candidate[3]))
                        if l.length() > 0 and all([not o.intersects(l) for o in objects ]):
                            if isAligned:
                                alignedAttachments.append(candidate)
                            else:
                                arbitraryAttachments.append(candidate)
    # randomly remove arbitrary attachments if there are too many
    if alignedAttachments != []:
        shuffle(arbitraryAttachments)
        arbitraryAttachments = arbitraryAttachments[:max(2,len(alignedAttachments))]
    return arbitraryAttachments + alignedAttachments

def samplePoint(objects):
    xs = list(set([ x for o in objects for x in o.usedXCoordinates() ]))
    ys = list(set([ y for o in objects for y in o.usedYCoordinates() ]))
    option = choice(range(3))
    if option == 0 and len(xs) > 0:
        return AbsolutePoint(Number(choice(xs)),Number(randomCoordinate()))
    if option == 1 and len(ys) > 0:
        return AbsolutePoint(Number(randomCoordinate()), Number(choice(ys)))
    return AbsolutePoint.sample()

def sampleCircle(objects):
    while True:
        p = samplePoint(objects)
        r = Number(1)
        c = Circle(p,r)
        if c.inbounds(): return c

def sampleRectangle(objects):
    while True:
        p1 = samplePoint(objects)
        p2 = samplePoint(objects)
        if p1.x != p2.x and p1.y != p2.y:
            x1 = Number(min([p1.x.n,p2.x.n]))
            x2 = Number(max([p1.x.n,p2.x.n]))
            y1 = Number(min([p1.y.n,p2.y.n]))
            y2 = Number(max([p1.y.n,p2.y.n]))
            p1 = AbsolutePoint(x1,y1)
            p2 = AbsolutePoint(x2,y2)
            return Rectangle(p1, p2)
                                

def sampleLine(attachedLines = []):
    concentration = 2.0
    if attachedLines != [] and random() < float(len(attachedLines))/(concentration + len(attachedLines)):
        (x1,y1,x2,y2) = choice(attachedLines)
        points = [AbsolutePoint(Number(x1),Number(y1)),AbsolutePoint(Number(x2),Number(y2))]
    elif random() < 1.0: # horizontal or vertical line: diagonals only allowed as attachments
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
    else: # arbitrary line between two points. Don't allow these.
        while True:
            p1 = AbsolutePoint.sample()
            p2 = AbsolutePoint.sample()
            if Line([p1,p2]).length() > 2:
                points = [p1,p2]
                break
    arrow = random() > 0.5
    if not arrow: # without an arrow there is no canonical orientation
        points = list(sorted(points, key = lambda p: (p.x.n,p.y.n)))
    return Line(points,
                solid = random() > 0.5,
                arrow = arrow)

def sampleWithoutIntersection(n, existingObjects, f):
    targetLength = len(existingObjects) + n
    while len(existingObjects) < targetLength:
        newObject = f()
        if not any([o.intersects(newObject) for o in existingObjects ]):
            existingObjects += [newObject]
    return existingObjects

def multipleObjects(rectangles = 0,lines = 0,circles = 0):
    def sampler():
        objects = []
        objects = sampleWithoutIntersection(circles, objects, lambda: sampleCircle(objects))
        objects = sampleWithoutIntersection(rectangles, objects, lambda: sampleRectangle(objects))
        attachedLines = proposeAttachmentLines(objects)
        objects = sampleWithoutIntersection(lines, objects, lambda: sampleLine(attachedLines))
        return Sequence(canonicalOrdering(objects))
    return sampler

def randomScene(maximumNumberOfObjects):
    def sampler():
        n = choice(range(maximumNumberOfObjects)) + 1

        shapeIdentities = [choice(range(3)) for _ in range(n) ]
        return multipleObjects(rectangles = len([x for x in shapeIdentities if x == 0 ]),
                               lines = len([x for x in shapeIdentities if x == 1 ]),
                               circles = len([x for x in shapeIdentities if x == 2 ]))()
    return sampler

def hiddenMarkovModel():
    spacing = 4
    offset = 2
    primitives = [ Circle(AbsolutePoint(Number((spacing + 2)*x + offset),
                                        Number((spacing + 2)*y + offset)), Number(1))
      for x in range(3)
      for y in range(2) ]
    # horizontal lines connecting hidden nodes
    primitives += [ Line.absolute(Number((spacing + 2)*x + offset + 1),
                                  Number(offset + y*(spacing + 2)),
                                  Number((spacing + 2)*x + offset + spacing + 1),
                                  Number(offset + y*(spacing + 2)),
                                  arrow = False)
                    for x in range(2)
                    for y in [1] ]
    primitives += [ Line.absolute(Number(offset + x*(spacing + 2)),
                                  Number((spacing + 2)*y + offset + spacing + 1),
                                  Number(offset + x*(spacing + 2)),
                                  Number((spacing + 2)*y + offset + 1),
                                  arrow = False)
                    for x in range(3)
                    for y in [0] ]
    return Sequence(primitives)
    
    

    
def icingModel():
    spacing = 3
    offset = 3
    primitives = [ Circle(AbsolutePoint(Number((spacing + 2)*x + offset),
                                        Number((spacing + 2)*y + offset)), Number(1))
      for x in range(3)
      for y in range(3) ]
    primitives += [ Line.absolute(Number(offset + x*(spacing + 2)),
                                  Number((spacing + 2)*y + offset + 1),
                                  Number(offset + x*(spacing + 2)),
                                  Number((spacing + 2)*y + offset + spacing + 1)
                                  )
                    for x in range(3)
                    for y in range(2) ]
    primitives += [ Line.absolute(Number((spacing + 2)*x + offset + 1),
                                  Number(offset + y*(spacing + 2)),
                                  Number((spacing + 2)*x + offset + spacing + 1),
                                  Number(offset + y*(spacing + 2)))
                    for x in range(2)
                    for y in range(3) ]
    return Sequence(primitives)
            

def handleGeneration(arguments):
    generators = {"individualCircle": multipleObjects(circles = 1),
                  "doubleCircleLine": multipleObjects(circles = 2,lines = 1),
                  "tripleLine": multipleObjects(lines = 3),
                  "doubleCircle": multipleObjects(circles = 2),
                  "randomScene": randomScene(8),
                  "tripleCircle": multipleObjects(circles = 3),
                  "individualRectangle": multipleObjects(rectangles = 1)}
    (n,startingPoint,k) = arguments
    # IMPORTANT!
    # You *do not* want directories with an enormous number of files in them
    # pack it all up into a directory that we will tar together later
    
    os.system('mkdir syntheticTrainingData/%d'%startingPoint)
    makeSyntheticData("syntheticTrainingData/%d/%s"%(startingPoint,n), generators[n], k = k, offset = startingPoint)
    print "Generated %d training sequences into syntheticTrainingData/%d"%(k,startingPoint)
    
if __name__ == '__main__':
    setCoordinateNoise(0.2)
    setRadiusNoise(0.1)

    if len(sys.argv) == 2 and sys.argv[1] == 'challenge':
        setCoordinateNoise(0.1)
        setRadiusNoise(0.05)
        x = render([hiddenMarkovModel().noisyTikZ()],showImage = True,yieldsPixels = True)[0]
        Image.fromarray(255*x).convert('L').save('challenge.png')
        assert False

    if len(sys.argv) == 2:
        totalNumberOfExamples = int(sys.argv[1])
    else:
        totalNumberOfExamples = 10000
    examplesPerBatch = totalNumberOfExamples/10 if totalNumberOfExamples > 100 else totalNumberOfExamples
    # this keeps any particular directory from getting too big
    if examplesPerBatch > 1000: examplesPerBatch = 1000
    
    os.system('rm -r syntheticTrainingData ; mkdir syntheticTrainingData')
    n = "randomScene"
    startingPoint = 0
    offsetsAndCounts = []
    while startingPoint < totalNumberOfExamples:
        kp = min(totalNumberOfExamples - startingPoint,examplesPerBatch)
        offsetsAndCounts.append((n,startingPoint,kp))
        startingPoint += examplesPerBatch
    print offsetsAndCounts
    workers = totalNumberOfExamples/examplesPerBatch
    if workers > 1:
        if workers > 15: workers = 15
        Pool(workers).map(handleGeneration, offsetsAndCounts)
    else:
        map(handleGeneration, offsetsAndCounts)

    if totalNumberOfExamples > 100:
        print "Generated files, building archive..."
        os.system('tar cvf syntheticTrainingData.tar -T /dev/null')

        for _,startingPoint,_ in offsetsAndCounts:
            os.system('cd syntheticTrainingData/%d && tar --append --file ../../syntheticTrainingData.tar . && cd ../..'%startingPoint)
            if totalNumberOfExamples > 100:
                os.system('rm -r syntheticTrainingData/%d'%startingPoint)

            os.system('rm -r syntheticTrainingData')
        print "Done. You should see everything in syntheticTrainingData.tar if you had at least 100 examples."
