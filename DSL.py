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

def addFeatures(fs):
    composite = {}
    for f in fs:
        for k in f:
            composite[k] = composite.get(k,0) + f[k]
    return composite

class Reflection():
    def __init__(self, command, body):
        self.command = command
        self.body = body
    def __str__(self):
        return "Reflection(%s,%s)"%(self.command,self.body)
    def convertToPython(self):
        return "%s(%s)"%(self.command, self.body.convertToPython())
    def extrapolations(self):
        for b in self.body.extrapolations():
            yield Reflection(self.command, b)
    def explode(self):
        return Reflection(self.command, self.body.explode())
    def features(self):
        return addFeatures([{'reflections':1,
                             'reflectionsX':int('x' in self.command),
                             'reflectionsY':int('y' in self.command)},
                            self.body.features()])
class Primitive():
    def __init__(self, k): self.k = k
    def __str__(self): return "Primitive(%s)"%self.k
    def convertToPython(self): return "[%s]"%self.k
    def extrapolations(self): yield self
    def explode(self):
        return self
    def features(self):
        return {'primitives':1,
                'lines':int('line' in self.k),
                'rectangle':int('rectangle' in self.k),
                'circles':int('circle' in self.k)}
class Loop():
    def __init__(self, v, bound, body, lowerBound = 0):
        self.v = v
        self.bound = bound
        self.body = body
        self.lowerBound = lowerBound
    def __str__(self):
        return "Loop(%s, %s, %s, %s)"%(self.v,self.lowerBound, self.bound,self.body)
    def convertToPython(self):
        return "[ _%s for %s in range(%s,%s) for _%s in %s ]"%(self.v,
                                                            self.v,
                                                            self.lowerBound,
                                                            self.bound,
                                                            self.v,
                                                            self.body.convertToPython())
    def extrapolations(self):
        for b in self.body.extrapolations():
            for ub,lb in [(1,1),(1,0),(0,1),(0,0)]:
                yield Loop(self.v, self.bound + ' + %d'%ub, b, lowerBound = self.lowerBound - lb)
    def explode(self):
        return Block([ Loop(self.v,self.bound,bodyExpression.explode(),lowerBound = self.lowerBound)
                       for bodyExpression in self.body.items ])
    def features(self):
        f2 = int(self.bound == '2')
        f3 = int(self.bound == '3')
        f4 = int(self.bound == '4')
        return addFeatures([{'loops':1,
                             '2': f2,
                             '3': f3,
                             '4': f4,
                             'variableLoopBound': int(f2 == 0 and f3 == 0 and f4 == 0)},
                            self.body.features()])                             
                
class Block():
    def convertToSequence(self):
        return Sequence([ p.evaluate() for p in eval(self.convertToPython()) ])
    def __init__(self, items): self.items = items
    def __str__(self): return "Block([%s])"%(", ".join(map(str,self.items)))
    def convertToPython(self):
        return " + ".join([ x.convertToPython() for x in self.items ])
    def extrapolations(self):
        if self.items == []: yield self
        else:
            for e in self.items[0].extrapolations():
                for s in Block(self.items[1:]).extrapolations():
                    yield Block([e] + s.items)
    def explode(self):
        return Block([ x.explode() for x in self.items ])
    def features(self):
        return addFeatures([ x.features() for x in self.items ])

# return something that resembles a syntax tree, built using the above classes
def sketchToDSL(trace, loopd = 0):
    accumulator = []
    lines = trace.split('\n')
    def depth(x):
        d = 0
        while d < len(x) and x[d] == ' ': d += 1
        return d
    j = 0
    while j < len(lines):
        l = lines[j]
        if 'rectangle' in l or 'circle' in l or 'line' in l:
            primitiveCommand = l.strip(' ')
            accumulator += [Primitive(primitiveCommand)]
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
            accumulator += [Reflection(reflectionCommand, body_)]
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
            j = body
            accumulator += [Loop(loopVariable, m.group(1), body_)]
        elif l == '': j += 1
        else:
            print l
            assert False
    return Block(accumulator)

def renderEvaluation(s, exportTo = None):
    parse = evaluate(eval(s))
    x0 = min([x for l in parse.lines for x in l.usedXCoordinates()  ])
    y0 = min([y for l in parse.lines for y in l.usedYCoordinates()  ])
    x1 = max([x for l in parse.lines for x in l.usedXCoordinates()  ])
    y1 = max([y for l in parse.lines for y in l.usedYCoordinates()  ])

    render([parse.TikZ()],showImage = exportTo == None,exportTo = exportTo,canvas = (x1+1,y1+1), x0y0 = (x0 - 1,y0 - 1))

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

