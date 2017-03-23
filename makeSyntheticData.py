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
            endingPoint = pixels[programPrefixes[j][k+1]]
            endingPoint.save("%s-%d-%d.png"%(filePrefix,j,k))
            
def canonicalOrdering(circles):
    # sort the circles so that there are always drawn in a canonical order
    return sorted(circles, key = lambda c: (c.center.x, c.center.y))

def doubleCircle():
    return Sequence(canonicalOrdering([Circle.sample(),Circle.sample()]))

def singleCircle():
    return Sequence([Circle.sample()])
        
makeSyntheticData(singleCircle, "syntheticTrainingData/individualCircle", 1000)
makeSyntheticData(doubleCircle, "syntheticTrainingData/doubleCircle", 1000)
