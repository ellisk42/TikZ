import os
from language import *
from render import render
from PIL import Image
import pickle

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
            startingPoint = pixels[programPrefixes[j][k]]
            endingPoint = pixels[programPrefixes[j][k+1]]
            startingPoint.save("%s-%d-%d-starting.png"%(filePrefix,j,k))
            endingPoint.save("%s-%d-%d-ending.png"%(filePrefix,j,k))
            


def doubleCircle():
    while True:
        c1 = Circle.sample()
        if c1.center.x < 5:
            break
    while True:
        c2 = Circle.sample()
        if c2.center.x > 6:
            break
    return Sequence([c1,c2])

def singleCircle():
    return Sequence([Circle.sample()])
        
makeSyntheticData(singleCircle, "syntheticTrainingData/individualCircle", 100)
makeSyntheticData(doubleCircle, "syntheticTrainingData/doubleCircle", 100)
