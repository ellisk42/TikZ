from render import render
from language import *
from utilities import *

import pickle
import os
import numpy as np

def precomputedRenderings():
    # exactly one rendering of a circle
    c = Sequence([Circle(AbsolutePoint(Number(2),Number(2)),Number(1))]).TikZ()
    c = 1.0 - render([c],yieldsPixels = True)[0]

    rectangles = {}
    for dx in range(1,15):
        for dy in range(1,15):
            rectangles[(dx,dy)] = Rectangle(AbsolutePoint(Number(1),Number(1)),
                                            AbsolutePoint(Number(1 + dx),Number(1 + dy))).TikZ()
    rectangleKeys = rectangles.keys()
    print "About to render %d rectangles"%(len(rectangleKeys))
    rectangleRenders = render([rectangles[k] for k in rectangleKeys ], yieldsPixels = True)
    for j,k in enumerate(rectangleKeys):
#        print k
        rectangles[k] = 1.0 - rectangleRenders[j]
        #        showImage(rectangles[k])

    lines = {} # keys: (dx,dy, solid, arrow). arrow is one of {None,True,False}. True = flip from canonical
    for dx in range(0,14 + 1):
        for dy in range(-14,14 + 1):
            if dx == 0 and dy == 0: continue

            if dy >= 0:
                y1 = 1
                y2 = y1 + dy
            else:
                y1 = 15
                y2 = y1 + dy            
            lines[(dx,dy,True,None)] = Line.absolute(Number(1), Number(y1),
                                                      Number(1 + dx), Number(y2)).TikZ()
            lines[(dx,dy,False,None)] = Line.absolute(Number(1), Number(y1),
                                                       Number(1 + dx), Number(y2),
                                                       solid = False).TikZ()
            p1 = AbsolutePoint(Number(1), Number(y1))
            p2 = AbsolutePoint(Number(1 + dx), Number(y2))
            # p1 > p2
            lines[(dx,dy,True,False)] = Line([p1,p2],arrow = True).TikZ()
            lines[(dx,dy,False,False)] = Line([p1,p2],arrow = True,solid = False).TikZ()
            #p2 > p1
            lines[(dx,dy,True,True)] = Line([p2,p1],arrow = True).TikZ()
            lines[(dx,dy,False,True)] = Line([p2,p1],arrow = True,solid = False).TikZ()
            
    print "About to render %d lines"%(len(lines))
    lineKeys = lines.keys()
    lineRenders = render([lines[k] for k in lineKeys ], yieldsPixels = True)
    for j,k in enumerate(lineKeys):
        lines[k] = 1.0 - lineRenders[j]
    return {'c':c,'r':rectangles,'l':lines}

globalFastRenderTable = None

def fastRender(program):
    loadPrecomputedRenderings()
    
    if isinstance(program,Sequence):
        output = sum([np.zeros((256,256))] + [fastRender(l) for l in program.lines ])
        output[output > 1] = 1
        return output

    if isinstance(program,Line):
        if program.length() < 0.1: return np.zeros((256,256))
        
        _x1 = program.points[0].x.n
        _y1 = program.points[0].y.n
        _x2 = program.points[1].x.n
        _y2 = program.points[1].y.n

        # canonical coordinates
        [(x1,y1),(x2,y2)] = sorted([(_x1,_y1),(_x2,_y2)])

        dx = x2 - x1
        dy = y2 - y1

        if program.arrow:
            if (_x1,_y1) < (_x2,_y2): # canonical
                arrow = False
            else:
                arrow = True
        else:
            arrow = None

        if not (dx,dy,program.solid,arrow) in globalFastRenderTable['l']:
            print program
        o = globalFastRenderTable['l'][(dx,dy,program.solid,arrow)]
        o = np.roll(o, (x1 - 1)*256/16,axis = 1)

        if dy >= 0: # starts at (1,1)
#            print "positive dy"
            return np.roll(o, 256 - (y1 - 1)*256/16,axis = 0)
        else:
#            print "negative dy"
            # starts at (1,14)
            return np.roll(o, (15 - y1)*256/16,axis = 0)

    if isinstance(program,Circle):
        x = program.center.x.n
        y = program.center.y.n
        dx = (x - 2)*256/16
        dy = 256 - (y - 2)*256/16
        o = np.roll(globalFastRenderTable['c'],dx,axis = 1)
        o = np.roll(o,dy,axis = 0)
        return o

    if isinstance(program,Rectangle):
        y1 = min([program.p1.y.n,program.p2.y.n])
        y2 = max([program.p1.y.n,program.p2.y.n])
        x1 = min([program.p1.x.n,program.p2.x.n])
        x2 = max([program.p1.x.n,program.p2.x.n])

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 or dy == 0: return np.zeros((256,256))
        if not (dx,dy) in globalFastRenderTable['r']:
            print program
            render([program.TikZ()],showImage = True)
        o = globalFastRenderTable['r'][(dx,dy)]
        o = np.roll(o, (x1 - 1)*256/16,axis = 1)
        o = np.roll(o, 256 - (y1 - 1)*256/16,axis = 0)
        return o
    assert False

def loadPrecomputedRenderings():
    global globalFastRenderTable
    if globalFastRenderTable != None: return
    
    if os.path.exists("precomputedRenderings.p"):
        globalFastRenderTable = pickle.load(open("precomputedRenderings.p",'rb'))
#        print "Loaded %d renderings."%(len(t['l']) + len(t['r']) + len(t['c']))
    else:
        globalFastRenderTable = precomputedRenderings()
        pickle.dump(globalFastRenderTable,open("precomputedRenderings.p",'wb'))


if __name__ == '__main__':
    for _ in range(20):
        p = Line.sample() #Circle(AbsolutePoint(Number(2),Number(3)),Number(1))
    #    p.solid = True
    #    p.arrow = False
        print p
        correct = (1 - render([Sequence([p]).TikZ()],yieldsPixels = True)[0])
        fast = (fastRender(p))
        showImage(correct)
        showImage(fast)
            # showImage(correct - fast)
            # print np.sum(np.abs(correct - fast))
