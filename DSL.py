from language import Rectangle,Circle,Line,AbsolutePoint,Number,Sequence
from utilities import *
from fastRender import fastRender
from render import render

import re

def reflectPoint(rx,ry,px,py):
    if rx != None: return (rx - px,py)
    if ry != None: return (px,ry - py)
    assert False
def reflect(x = None,y = None):
    def reflector(stuff):
        return stuff + [ o.reflect(x = x,y = y) for o in stuff ]
    return reflector
    

class line():
    def __init__(self, x1, y1, x2, y2, arrow = None, solid = None):
        self.arrow = arrow
        self.solid = solid
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    def evaluate(self):
        return Line.absoluteNumbered(self.x1,
                                     self.y1,
                                     self.x2,
                                     self.y2,
                                     arrow = self.arrow,
                                     solid = self.solid)
    def reflect(self, x = None,y = None):
        (x1,y1) = reflectPoint(x,y,self.x1,self.y1)
        (x2,y2) = reflectPoint(x,y,self.x2,self.y2)
        if self.arrow:
            return line(x1,y1,x2,y2,arrow = True,solid = self.solid)
        else:
            (a,b) = min((x1,y1),(x2,y2))
            (c,d) = max((x1,y1),(x2,y2))
            return line(a,b,c,d,
                        arrow = False,
                        solid = self.solid)
        

class rectangle():
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    def evaluate(self):
        return Rectangle.absolute(self.x1,self.y1,self.x2,self.y2)
    
    def reflect(self, x = None,y = None):
        (x1,y1) = reflectPoint(x,y,self.x1,self.y1)
        (x2,y2) = reflectPoint(x,y,self.x2,self.y2)
        return rectangle(min(x1,x2),
                         min(y1,y2),
                         max(x1,x2),
                         max(y1,y2))

class circle():
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def evaluate(self):
        return Circle(center = AbsolutePoint(Number(self.x),Number(self.y)),
                      radius = Number(1))
    def reflect(self, x = None,y = None):
        return circle(*reflectPoint(x,y,self.x,self.y))

def evaluate(stuff):
    return Sequence([o.evaluate() for o in stuff ])


def sketchToDSL(trace, loopd = 0):
    accumulator = ''
    lines = trace.split('\n')
    def depth(x):
        d = 0
        while d < len(x) and x[d] == ' ': d += 1
        return d
    j = 0
    while j < len(lines):
        l = lines[j]
        if 'rectangle' in l or 'circle' in l or 'line' in l:
            primitiveCommand = '[%s]'%l
            if accumulator != '' and accumulator[-1] == ']':
                primitiveCommand = ' + %s'%primitiveCommand
            accumulator += primitiveCommand
            j += 1
        elif 'reflect' in l:
            m = re.search('([xy]) = ([0-9]+)',l)
            if m == None:
                print l
                assert False
            reflectionCommand = 'reflect(%s = %s)'%(m.group(1),m.group(2))
            depthThreshold = depth(lines[j])
            body = j
            while body < len(lines) and depth(lines[body]) >= depthThreshold:
                body += 1
            body_ = sketchToDSL("\n".join(lines[(j+1):body]),loopd)
            if accumulator != '': accumulator += ' + '
            accumulator += "%s(%s)"%(reflectionCommand,body_)
            j = body
        elif 'for' in l:
            m = re.search('\((.*)\)',l)
            if m == None:
                print l
                assert False
            loopVariable = ['i','j'][loopd]
            depthThreshold = depth(lines[j])
            body = j+1
            while body < len(lines) and depth(lines[body]) > depthThreshold:
                body += 1
            
            body_ = sketchToDSL("\n".join(lines[(j+1):body]),loopd+1)
            loopCommand = '[ _%s for %s in range(%s) for _%s in %s ]'%(loopVariable,
                                                                       loopVariable,
                                                                       m.group(1),
                                                                       loopVariable,
                                                                       body_)
            j = body
            if accumulator != '' and accumulator[-1] == ']':
                accumulator += ' + '
            accumulator += loopCommand
        elif l == '': j += 1
        else:
            print l
            assert False
    return accumulator

if __name__ == '__main__':
    sequence = evaluate(eval(sketchToDSL('''
      circle(9,1)
        for (4)
            circle(2*i + 3,-3*i + 10)
            circle(-2*i + 5,3*i + 1)
            line(2*i + 2,-3*i + 7,2*i + 3,-3*i + 9,arrow = False,solid = True)
            line(-2*i + 7,3*i + 3,-2*i + 8,3*i + 1,arrow = False,solid = True)
    ''')))

    #    [  line(0,1,0,4,arrow = False,solid = True)] + [  rectangle(2,0,5,3)] + [ _i for i in range(2)) for _i in [        line(0,3*i + 1,2,3*i,arrow = False,solid = True)] + [        line(-3*i + 3,4,-2*i + 5,3,arrow = False,solid = True)] ]
    # expression = reflect(y = 5)([circle(1,9),
    #                              line(1,2,1,4,True,True)])
    render([sequence.TikZ()],showImage = True)

