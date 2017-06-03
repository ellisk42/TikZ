from utilities import *

import pickle
from random import choice
import os

def loadSamples(d):
    names = [str(j) for j in range(10) ] + [chr(ord('A') + j) for j in range(26) ] + [chr(ord('a') + j) for j in range(26) ]

    namesWeCareAbout = ['A','B','C','X','Y','Z']

    dx = 60/2
    dy = 45/2

    examples = {}
    for j,n in enumerate(names):
        if not n in namesWeCareAbout: continue
        
        examples[n] = []
        for f in picturesInDirectory(d + '/Sample%03d'%(j+1)):
            i = Image.open(f).resize((dx,dy),
                                     Image.BICUBIC)
            examples[n].append(image2array(i))

    if True:
        for n in namesWeCareAbout:
            print n
            showImage(examples[n][0])
    pickle.dump(examples,open('characters.p','wb'))

CHARACTERMAP = None
def loadCharacters():
    global CHARACTERMAP
    if CHARACTERMAP == None:
        with open('characters.p','rb') as h:
            CHARACTERMAP = pickle.load(h)

def sampleCharacter(c):
    loadCharacters()
    return choice(CHARACTERMAP[c])

def blitCharacter(surface,x,y,c):
    (x,y) = (int(y),int(x))
    c = sampleCharacter(c)
    (dx,dy) = c.shape
    (w,h) = surface.shape
    x = h - x
    x1 = max(x - dx/2,0)
    x2 = min(x + dx/2,w)
    y1 = max(y - dy/2,0)
    y2 = min(y + dy/2,h)
    
    surface[x1:x2,y1:y2] += c[x1 - (x - dx/2) : x1 - (x - dx/2) + (x2 - x1),
                              y1 - (y - dy/2) : y1 - (y - dy/2) + (y2 - y1)]

if __name__ == '__main__':
    loadSamples('characters')
