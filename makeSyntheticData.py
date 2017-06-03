from multiprocessing import Pool
import sys
import os
from language import *
from render import render
from character import *
from PIL import Image
import pickle
from random import choice,shuffle
from utilities import showImage

CANONICAL = True

def makeSyntheticData(filePrefix, sample, k = 1000, offset = 0):
    """sample should return a program"""
    programs = [sample() for _ in range(k)]
    print "Sampled %d programs."%k
    # remove all of the labels - we will render them separately
    noLabels = [ Sequence([ l for l in p.lines if not isinstance(l,Label) ])
                 for p in programs ]
    onlyLabels = [ [ l for l in p.lines if isinstance(l,Label) ]
                   for p in programs ]
    noisyTargets = [ p.noisyTikZ() for p in noLabels ]
    distinctPrograms = list(set(noisyTargets))
    pixels = render(distinctPrograms, yieldsPixels = True)
    print "Rendered %d images for %s."%(len(distinctPrograms),filePrefix)

    #pixels = [ Image.fromarray(ps*255).convert('L') for ps in pixels ]
    pixels = dict(zip(distinctPrograms,pixels))
    
    for j in range(len(programs)):
        pickle.dump(programs[j], open("%s-%d.p"%(filePrefix,j + offset),'wb'))
        unlabeledPixels = 1 - pixels[noisyTargets[j]]
        for l in onlyLabels[j]:
            blitCharacter(unlabeledPixels,
                          l.p.x*16,l.p.y*16,
                          l.c)
        unlabeledPixels[unlabeledPixels > 1] = 1
        unlabeledPixels = (1 - unlabeledPixels)*255
        labeledPixels = Image.fromarray(unlabeledPixels).convert('L')
        labeledPixels.save("%s-%d-noisy.png"%(filePrefix,j + offset))

        if False:
            Image.fromarray(255*programs[j].draw()).convert('L').save("%s-%d-clean.png"%(filePrefix,j + offset))
            
def canonicalOrdering(things):
    if things == [] or not CANONICAL: return things
    cs = [c for c in things if isinstance(c,Circle) ]
    cs = sorted(cs, key = lambda c: (c.center.x, c.center.y, -c.radius))
    rs = [c for c in things if isinstance(c,Rectangle) ]
    rs = sorted(rs, key = lambda r: (r.p1.x,
                                     r.p1.y))
    ls = [c for c in things if isinstance(c,Line) ]
    ls = sorted(ls, key = lambda l: (l.points[0].x,l.points[0].y,
                                     l.points[1].x,l.points[1].y,
                                     l.solid,l.arrow))
    ts = [t for t in things if isinstance(t,Label) ]
    ts = sorted(ts, key = lambda t: (t.p.x,t.p.y))
    return cs + rs + ls + ts


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
                        l = Line.absolute((candidate[0]),
                                          (candidate[1]),
                                          (candidate[2]),
                                          (candidate[3]))
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
    ls = [ p for o in objects for p in (o.points if isinstance(o,Line) else []) ]
    xs = list(set([ x for o in objects for x in o.usedXCoordinates() ]))
    ys = list(set([ y for o in objects for y in o.usedYCoordinates() ]))
    if ls != [] and random() < 0.1: return choice(ls)
    option = choice(range(3))
    if option == 0 and len(xs) > 0:
        return AbsolutePoint((choice(xs)),(randomCoordinate()))
    if option == 1 and len(ys) > 0:
        return AbsolutePoint((randomCoordinate()), (choice(ys)))
    return AbsolutePoint.sample()


def sampleCircle(objects):
    existingRadii = [ c.radius for c in objects if isinstance(c,Circle) ]
    concentrationParameter = 1.5
    reuseProbability = len(existingRadii)/(len(existingRadii) + concentrationParameter)
    while True:
        p = samplePoint(objects)
        if random() < reuseProbability:
            r = choice(existingRadii)
        else:
            r = sampleRadius()
        c = Circle(p,r)
        if c.inbounds(): return c

def sampleRectangle(objects):
    while True:
        p1 = samplePoint(objects)
        p2 = samplePoint(objects)
        if p1.x != p2.x and p1.y != p2.y:
            x1 = (min([p1.x,p2.x]))
            x2 = (max([p1.x,p2.x]))
            y1 = (min([p1.y,p2.y]))
            y2 = (max([p1.y,p2.y]))
            if x2 - x1 > 0.5 and y2 - y1 > 0.5:
                p1 = AbsolutePoint(x1,y1)
                p2 = AbsolutePoint(x2,y2)
                return Rectangle(p1, p2)

def sampleLabel(objects):
    l = Label.sample()
    l.p = samplePoint(objects)
    return l
                                

def sampleLine(objects, attachedLines = []):
    concentration = 2.0
    if attachedLines != [] and random() < float(len(attachedLines))/(concentration + len(attachedLines)):
        (x1,y1,x2,y2) = choice(attachedLines)
        points = [AbsolutePoint((x1),(y1)),AbsolutePoint((x2),(y2))]
    elif random() < 1.0: # horizontal or vertical line: diagonals only allowed as attachments
        p1 = samplePoint(objects)
        x1 = p1.x
        y1 = p1.y
        if choice([True,False]):
            # x1 == x2; y1 != y2
            y2 = y1
            while abs(y2 - y1) < 1: y2 = randomCoordinate()
            points = [AbsolutePoint((x1),(y1)),AbsolutePoint((x1),(y2))]
        else:
            # x1 != x2; y1 == y2
            x2 = x1
            while abs(x2 - x1) < 1: x2 = randomCoordinate()
            points = [AbsolutePoint((x1),(y1)),AbsolutePoint((x2),(y1))]
    else: # arbitrary line between two points. Don't allow these.
        while True:
            p1 = AbsolutePoint.sample()
            p2 = AbsolutePoint.sample()
            if Line([p1,p2]).length() > 2:
                points = [p1,p2]
                break
    arrow = random() > 0.5
    if not arrow: # without an arrow there is no canonical orientation
        points = list(sorted(points, key = lambda p: (p.x,p.y)))
    return Line(points,
                solid = random() > 0.1,
                arrow = arrow)

def sampleWithoutIntersection(n, existingObjects, f):
    targetLength = len(existingObjects) + n
    maximumAttempts = 10000
    attemptsSoFar = 0
    while len(existingObjects) < targetLength:
        attemptsSoFar += 1
        if attemptsSoFar > maximumAttempts:
            break
        
        newObject = f()
        if not any([o.intersects(newObject) for o in existingObjects ]):
            existingObjects += [newObject]
    return existingObjects

def multipleObjects(rectangles = 0,lines = 0,circles = 0,labels = 0):
    def sampler():
        objects = []
        objects = sampleWithoutIntersection(circles, objects, lambda: sampleCircle(objects))
        objects = sampleWithoutIntersection(rectangles, objects, lambda: sampleRectangle(objects))
        objects = sampleWithoutIntersection(labels, objects, lambda: sampleLabel(objects))
        attachedLines = proposeAttachmentLines(objects)
        objects = sampleWithoutIntersection(lines, objects, lambda: sampleLine(objects,attachedLines))
        return Sequence(canonicalOrdering(objects))
    return sampler

def randomScene(maximumNumberOfObjects):
    def sampler():
        while True:
            n = choice(range(maximumNumberOfObjects)) + 1
            shapeIdentities = [choice(range(4)) for _ in range(n) ]
            numberOfLabels = len([i for i in shapeIdentities if i == 3 ])
            # make it so that there are not too many labels
            if numberOfLabels > n/2: continue
            
            return multipleObjects(rectangles = len([x for x in shapeIdentities if x == 0 ]),
                                   lines = len([x for x in shapeIdentities if x == 1 ]),
                                   circles = len([x for x in shapeIdentities if x == 2 ]),
                                   labels = len([x for x in shapeIdentities if x == 3 ]))()
    return sampler

def handleGeneration(arguments):
    generators = {"randomScene": randomScene(12)}
    (n,startingPoint,k) = arguments
    # IMPORTANT!
    # You *do not* want directories with an enormous number of files in them
    # pack it all up into a directory that we will tar together later
    
    os.system('mkdir %s/%d'%(outputName,startingPoint))
    makeSyntheticData("%s/%d/%s"%(outputName,startingPoint,n), generators[n], k = k, offset = startingPoint)
    print "Generated %d training sequences into %s/%d"%(k,outputName,startingPoint)
    
if __name__ == '__main__':
    loadCharacters()

    if 'continuous' in sys.argv:
        setSnapToGrid(False)
        outputName = 'syntheticContinuousTrainingData'
    else:
        # models imperfect grid alignment
        setCoordinateNoise(0.2)
        setRadiusNoise(0.1)
        outputName = 'syntheticTrainingData'

    if len(sys.argv) > 1:
        totalNumberOfExamples = int(sys.argv[1])
    else:
        totalNumberOfExamples = 100000
    examplesPerBatch = totalNumberOfExamples/10 if totalNumberOfExamples > 100 else totalNumberOfExamples
    # this keeps any particular directory from getting too big
    if examplesPerBatch > 1000: examplesPerBatch = 1000
    
    os.system('rm -r %s %s.tar ; mkdir %s'%(outputName,outputName,outputName))
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
        if workers > 10: workers = 10
        Pool(workers).map(handleGeneration, offsetsAndCounts)
    else:
        map(handleGeneration, offsetsAndCounts)

    if totalNumberOfExamples > 100:
        print "Generated files, building archive..."
        os.system('tar cvf %s.tar -T /dev/null'%outputName)

        for _,startingPoint,_ in offsetsAndCounts:
            os.system('cd %s/%d && tar --append --file ../../%s.tar . && cd ../..'%(outputName,startingPoint,outputName))
        #     if totalNumberOfExamples > 100:
        #         os.system('rm -r syntheticTrainingData/%d'%startingPoint)

        # os.system('rm -r syntheticTrainingData')
        print "Done. You should see everything in %s.tar if you had at least 100 examples."%(outputName)
