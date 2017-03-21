import os
from language import *
from render import render
from PIL import Image
import pickle

def makeSyntheticData(sample, prefix, k = 1000):
    programs = [sample() for _ in range(k)]
    pixels = render(map(str, programs), yieldsPixels = True)
    for j in range(k):
        image = Image.fromarray(pixels[j]*255).convert('L')
        image.save("%s-%d.png"%(prefix,j))
        pickle.dump(programs[j], open("%s-%d.p"%(prefix,j),'wb'))


makeSyntheticData(Circle.sample, "syntheticTrainingData/individualCircle", 1000) 
