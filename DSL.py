from language import Rectangle,Circle,Line,AbsolutePoint,Sequence
from utilities import *
from fastRender import fastRender
from render import render

import re
import itertools as iterationTools
import numpy as np
import random

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
        return Line.absolute(self.x1,
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
        return Circle(center = AbsolutePoint(self.x,self.y),
                      radius = 1)
    def reflect(self, x = None,y = None):
        return circle(*reflectPoint(x,y,self.x,self.y))

def addFeatures(fs):
    composite = {}
    for f in fs:
        for k in f:
            composite[k] = composite.get(k,0) + f[k]
    return composite

class AbstractionVariable():
    def __init__(self,v): self.v = v
    def __eq__(self,o): return isinstance(o,AbstractionVariable) and self.v == o.v
    def __str__(self): return "__v__(%d)"%(self.v)
class Environment():
    def __init__(self,b = []): self.bindings = b
    def lookup(self,x):
        for k,v in self.bindings:
            if k == x: return v
        return None
    def extend(self,x,y = None):
        if y == None: x,y = AbstractionVariable(len(self.bindings)),x
        return Environment(self.bindings + [(x,y)]), x
    def __str__(self):
        return "Environment([%s])"%(", ".join([ "%s -> %s"%(k,v) for k,v in self.bindings ]))
    def makeVariableWithValue(self,z):
        for k,v in self.bindings:
            if v == z: return k,self
        e,v = self.extend(z)
        return v,e
    def abstractConstant(self,n,m):
        if n == m: return (m,self)
        return self.makeVariableWithValue((n,m))
        
    def getTypes(self):
        t = []
        for _,(x,y) in self.bindings:
            if isinstance(x,int):
                assert isinstance(y,int)
                t.append(int)
            elif x in ['x','y']:
                assert y in ['x','y']
                t.append(str)
        return t
    def randomInstantiation(self):
        b = []
        for v,(x,y) in self.bindings:
            if isinstance(x,int):
                assert isinstance(y,int)
                average = (x + y)/2.0
                standardDeviation = (((x - average)**2 + (y - average)**2)/2.0)**(0.5)
                z = int(np.random.normal(loc = average,
                                         scale = standardDeviation))
                if z == 0 and x != 0 and y != 0:
                    z = random.choice([-1,1])                
                b.append((v,z))
            elif x in ['x','y']:
                assert y in ['x','y']
                b.append((v,random.choice(['x','y'])))
        return Environment(b)
    def firstInstantiation(self):
        return Environment([(v,x) for v,(x,y) in self.bindings ])
    def secondInstantiation(self):
        return Environment([(v,y) for v,(x,y) in self.bindings ])
    
class AbstractionFailure(Exception):
    pass

class LinearExpression():
    def __init__(self, m,x,b):
        self.m = m
        self.x = x
        self.b = b
        if x == None: assert m == 0
        if isinstance(m,int) and m == 0: assert x == None
    def __str__(self):
        if self.x == None: return str(self.b)
        return '(%s*%s + %s)'%(self.m,self.x,self.b)
    def pretty(self):
        if self.m == 0: return str(self.b)
        if self.b == 0:
            return "%s * %s"%(self.m,self.x)
        else:
            return "%s * %s + %s"%(self.m,self.x,self.b)
    def __eq__(self,o): return isinstance(o,LinearExpression) and self.m == o.m and self.x == o.x and self.b == o.b

    def freeVariables(self):
        if x == None: return []
        return [x]
    
    def abstract(self,other,e):
        if self.x != other.x: raise AbstractionFailure('')
        
        def abstractNumber(n,m,e_):
            if n == m: return (n,e_)
            for k,(n_,m_) in e_.bindings:
                if n_ == n and m_ == m: return (k,e_)
            e_,newVariable = e_.extend((n,m))
            return (newVariable,e_)
        m,e = abstractNumber(self.m,other.m,e)
        b,e = abstractNumber(self.b,other.b,e)
        return LinearExpression(m,self.x,b),e

    def substitute(self,e):
        m = e.lookup(self.m)
        if m == None: m = self.m
        b = e.lookup(self.b)
        if b == None: b = self.b
        return LinearExpression(m,self.x,b)

class RelativeExpression():
    orientations = {'n':'North',
                    'w':'West',
                    'e':'East',
                    's':'South'}
    def __init__(self, index, orientation):
        self.index = index
        assert orientation in RelativeExpression.orientations.keys()
        self.orientation = orientation
    def __str__(self):
        return '%s(%d)'%(RelativeExpression.orientations[self.orientation],self.index)
    def pretty(self): return str(self)
    def __eq__(self,o):
        return isinstance(o,RelativeExpression) and o.index == self.index and o.orientation == self.orientation
    def abstract(self,other,e):
        index,e = e.abstractConstant(self.index, other.index)
        orientation,e = e.abstractConstant(self.orientation, other.orientation)
        return RelativeExpression(index, orientation),e

    def substitute(self,e):
        raise Exception('not implemented: relative expression substitution')

                

class Primitive():
    def pretty(self):
        p = '%s(%s)'%(self.k,",".join([ (a if isinstance(a,str) else a.pretty())
                                        for a in self.arguments]))
        return p.replace(',arrow',',\narrow')
    def __init__(self, k, *arguments):
        self.k = k
        self.arguments = list(arguments)
    def __str__(self): return "Primitive(%s)"%(", ".join(map(str,[self.k] + self.arguments)))
    def hoistReflection(self):
        return
        yield
    def convertToPython(self): return "[%s(%s)]"%(self.k,", ".join(map(str,self.arguments)))
    def extrapolations(self): yield self
    def explode(self):
        return self
    def features(self):
        return {'primitives':1,
                'lines':int('line' in self.k),
                'rectangle':int('rectangle' in self.k),
                'circles':int('circle' in self.k)}
    def rewrites(self):
        return
        yield

    def removeDeadCode(self): return self

    def mapExpression(self,l):
        return Primitive(self.k, *[ (l(a) if isinstance(a,LinearExpression) else a) for a in self.arguments ])

    def walk(self):
        yield self

    def cost(self): return 1

    def abstract(self,other,e):
        if isinstance(other,Primitive) and self.k == other.k:
            arguments = []
            for p,q in zip(self.arguments, other.arguments):
                if isinstance(p,LinearExpression):
                    assert isinstance(q,LinearExpression)
                    a,e = p.abstract(q,e)
                    arguments.append(a)
                else: arguments.append(p)
            return Primitive(self.k,*arguments),e
        raise AbstractionFailure('different primitives')

    def substitute(self,e):
        return Primitive(self.k, *[ (a.substitute(e) if isinstance(a,LinearExpression) else a) for a in self.arguments ])

    def depth(self): return 1


class Reflection():
    def pretty(self):
        return "reflect(%s = %s){\n%s\n}"%(self.axis,self.coordinate,self.body.pretty())
    def __init__(self, axis, coordinate, body):
        self.axis = axis
        self.coordinate = coordinate
        self.body = body
    def removeDeadCode(self):
        body = self.body.removeDeadCode()
        return Reflection(self.axis, self.coordinate, body)
            
    def hoistReflection(self):
        for j,p in enumerate(self.body.items):
            if isinstance(p,Primitive):
                newBlock = list(self.body.items)
                del newBlock[j]
                newBlock = Block(newBlock)
                yield Block([p,Reflection(self.axis,self.coordinate,newBlock)])
                
    def __str__(self):
        return "Reflection(%s,%s,%s)"%(self.axis, self.coordinate,self.body)
    def convertToPython(self):
        return "reflect(%s = %s)(%s)"%(self.axis, self.coordinate, self.body.convertToPython())
    def extrapolations(self):
        for b in self.body.extrapolations():
            yield Reflection(self.axis, self.coordinate, b)
    def explode(self):
        return Reflection(self.axis, self.coordinate, self.body.explode())
    def features(self):
        return addFeatures([{'reflections':1,
                             'reflectionsX':int('x' == self.axis),
                             'reflectionsY':int('y' == self.axis)},
                            self.body.features()])
    def rewrites(self):
        for b in self.body.rewrites():
            yield Reflection(self.axis,self.coordinate,b)

    def mapExpression(self,l):
        return Reflection(self.axis, self.coordinate, self.body.mapExpression(l))

    def walk(self):
        yield self
        for w in self.body.walk():
            yield w

    def cost(self):
        return 1 + self.body.cost()

    def abstract(self,other,e):
        if not isinstance(other, Reflection): raise AbstractionFailure('abstracting a reflection with a not reflection')
        if self.axis == other.axis: axis = self.axis
        else: axis,e = e.makeVariableWithValue((self.axis,other.axis))
        if self.coordinate == other.coordinate: coordinate = self.coordinate
        else: coordinate,e = e.makeVariableWithValue((self.coordinate, other.coordinate))
        body,e = self.body.abstract(other.body,e)
        return Reflection(axis, coordinate, body),e

    def substitute(self,e):
        axis = e.lookup(self.axis)
        if axis == None: axis = self.axis
        coordinate = e.lookup(self.coordinate)
        if coordinate == None: coordinate = self.coordinate
        return Reflection(axis, coordinate, self.body.substitute(e))

    def depth(self): return 1 + self.body.depth()


class Loop():
    def pretty(self):
        p = "for (%s < %s){\n"%(self.v,self.bound)
        if self.boundary != None:
            p += "if (%s > 0){\n%s\n}\n"%(self.v,self.boundary.pretty())
        p += "%s\n}"%(self.body.pretty())
        return p
    def __init__(self, v, bound, body, boundary = None, lowerBound = 0):
        self.v = v
        self.bound = bound
        self.body = body
        self.boundary = boundary
        self.lowerBound = lowerBound
    def removeDeadCode(self):
        body = self.body.removeDeadCode()
        boundary = self.boundary.removeDeadCode() if self.boundary != None else None
        return Loop(self.v, self.bound, body, boundary = boundary, lowerBound = self.lowerBound)
    def hoistReflection(self):
        for h in self.body.hoistReflection():
            yield Loop(self.v,self.bound,h,boundary = self.boundary,lowerBound = self.lowerBound)
        if self.boundary != None:
            for h in self.boundary.hoistReflection():
                yield Loop(self.v,self.bound,self.body,boundary = h,lowerBound = self.lowerBound)
                
    def __str__(self):
        if self.boundary != None:
            return "Loop(%s, %s, %s, %s, boundary = %s)"%(self.v,self.lowerBound, self.bound,self.body,self.boundary)
        return "Loop(%s, %s, %s, %s)"%(self.v,self.lowerBound, self.bound,self.body)
    def convertToPython(self):
        body = self.body.convertToPython()
        if self.boundary != None:
            body += " + ((%s) if %s > %s else %s)"%(self.boundary.convertToPython(),
                                                    self.v,
                                                    self.lowerBound,
                                                    '[]')
            
        return "[ _%s for %s in range(%s,%s) for _%s in (%s) ]"%(self.v,
                                                               self.v,
                                                               self.lowerBound,
                                                               self.bound,
                                                               self.v,
                                                               body)
        
    def extrapolations(self):
        for b in self.body.extrapolations():
            for boundary in ([None] if self.boundary == None else self.boundary.extrapolations()):
                for ub,lb in [(1,1),(1,0),(0,1),(0,0)]:
                    yield Loop(self.v, '%s + %d'%(self.bound,ub), b,
                               lowerBound = self.lowerBound - lb,
                               boundary = boundary)
    def explode(self):
        shrapnel = [ Loop(self.v,self.bound,bodyExpression.explode(),lowerBound = self.lowerBound)
                       for bodyExpression in self.body.items ]
        if self.boundary != None:
            shrapnel += [ Loop(self.v,self.bound,Block([]),lowerBound = self.lowerBound,
                               boundary = bodyExpression.explode())
                       for bodyExpression in self.boundary.items ]
        return Block(shrapnel)
    def features(self):
        f2 = int(str(self.bound) == '2')
        f3 = int(str(self.bound) == '3')
        f4 = int(str(self.bound) == '4')
        return addFeatures([{'loops':1,
                             '2': f2,
                             '3': f3,
                             '4': f4,
                             'boundary': int(self.boundary != None),
                             'variableLoopBound': int(f2 == 0 and f3 == 0 and f4 == 0)},
                            self.body.features(),
                            self.boundary.features() if self.boundary != None else {}])
    def rewrites(self):
        for b in self.body.rewrites():
            yield Loop(self.v, self.bound, b, self.boundary, self.lowerBound)
        if self.boundary != None:
            for b in self.boundary.rewrites():
                yield Loop(self.v, self.bound, self.body, b, self.lowerBound)
    def mergeWithOtherLoop(self,other):
        assert self.v == other.v and self.lowerBound == other.lowerBound
        boundary = self.boundary
        if other.boundary != None:
            if boundary == None: boundary = other.boundary
            else: boundary = Block(self.boundary.items + other.boundary.items)
        body = Block(self.body.items + other.body.items)
        return Loop(self.v, self.bound, body, boundary, self.lowerBound)
    def mergeWithOtherLoopDifferentBounds(self,other):
        assert self.bound.b == other.bound.b + 1
        assert self.v == other.v and self.lowerBound == other.lowerBound
        assert other.boundary == None

        boundary = other.body.mapExpression(lambda l: LinearExpression(l.m,l.x,l.b-l.m) if l.x == self.v else l).items
        if self.boundary != None: boundary = boundary + self.boundary.items

        return Loop(self.v, self.bound, self.body, Block(boundary), self.lowerBound)


    def mapExpression(self,l):
        return Loop(self.v, l(self.bound), self.body.mapExpression(l),
                    None if self.boundary == None else self.boundary.mapExpression(l),
                    self.lowerBound)

    def walk(self):
        yield self
        for x in self.body.walk(): yield x
        if self.boundary != None:
            for x in self.boundary.walk(): yield x

    def cost(self):
        cost = self.body.cost()
        if self.boundary != None:
            cost += self.boundary.cost()
        if self.bound.m == 0 and self.bound.b == 2: cost += 1
        return cost + 1

    def abstract(self,other,e):
        if not isinstance(other,Loop) or self.v != other.v or ((other.boundary == None) != (self.boundary == None)):
            raise AbstractionFailure('Loop abstraction')
        assert self.lowerBound == 0 and other.lowerBound == 0
        bound,e = self.bound.abstract(other.bound,e)
        body,e = self.body.abstract(other.body,e)
        if self.boundary == None: boundary = None
        else: boundary,e = self.boundary.abstract(other.boundary,e)
        return Loop(self.v,bound, body, boundary),e

    def substitute(self,e):
        return Loop(self.v,
                    self.bound.substitute(e),
                    self.body.substitute(e),
                    None if self.boundary == None else self.boundary.substitute(e),
                    self.lowerBound)

    def depth(self): return 1 + max(self.body.depth(),
                                    0 if self.boundary == None else self.boundary.depth())

                
class Block():
    def pretty(self): return ";\n".join([x.pretty() for x in self.items ])
    def convertToSequence(self):
        return Sequence([ p.evaluate() for p in eval(self.convertToPython()) ])
    def __init__(self, items): self.items = items
    def __str__(self): return "Block([%s])"%(", ".join(map(str,self.items)))
    def convertToPython(self):
        if self.items == []: return "[]"
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
    def hoistReflection(self):
        for j,x in enumerate(self.items):
            for y in x.hoistReflection():
                copy = list(self.items)
                copy[j] = y
                yield Block(copy)

    def fixReflections(self,target):
        distance = self.convertToSequence() - target
        if distance == 0: return self

        print "Fixing reflections"

        candidates = [self] + list(self.hoistReflection())
        sequences = [k.convertToSequence() for k in candidates ]
        distances = [target - s for s in sequences ]
        best = min(range(len(distances)),key = lambda k: distances[k])
        if distances[best] == distance: return self
        return candidates[best].fixReflections(target)

    def removeDeadCode(self):
        items = [ x.removeDeadCode() for x in self.items ]
        keptItems = []
        for x in items:
            if isinstance(x,Reflection) and x.body.items != []: keptItems.append(x)
            elif isinstance(x,Loop) and (x.body.items != [] or (x.boundary != None and x.boundary.items != [])):
                keptItems.append(x)
            elif isinstance(x,Primitive):
                keptItems.append(x)
        return Block(keptItems)

    def rewriteUpToDepth(self,d):
        rewrites = [[self]]
        for _ in range(d):
            rewrites.append([ r for b in rewrites[-1]
                              for r in b.rewrites() ])
        return [ r for rs in rewrites for r in rs  ]

    def rewrites(self):
        for j,x in enumerate(self.items):
            for r in x.rewrites():
                newItems = list(self.items)
                newItems[j] = r
                yield Block(newItems)
        for j,x in enumerate(self.items):
            for k,y in enumerate(self.items):
                if not (j < k): continue
                if isinstance(x,Loop) and isinstance(y,Loop):
                    if x.bound == y.bound and x.lowerBound == y.lowerBound:
                        newLoop = x.mergeWithOtherLoop(y)
                        newItems = list(self.items)
                        newItems[j] = newLoop
                        del newItems[k]
                        yield Block(newItems)
                    if x.bound.m == y.bound.m and x.bound.x == y.bound.x and abs(x.bound.b - y.bound.b) == 1:
                        # make it so that y is the one with fewer iterations
                        if x.bound.b < y.bound.b: x,y = y,x
                        if y.boundary != None: continue
                        newLoop = x.mergeWithOtherLoopDifferentBounds(y)
                        newItems = list(self.items)
                        newItems[j] = newLoop
                        del newItems[k]
                        yield Block(newItems)
                if isinstance(x,Reflection) and isinstance(y,Reflection):
                    if (x.axis,x.coordinate) == (y.axis,y.coordinate):
                        newItems = list(self.items)
                        newItems[j] = Reflection(x.axis, x.coordinate, Block(x.body.items + y.body.items))
                        del newItems[k]
                        yield Block(newItems)

    def mapExpression(self,l):
        return Block([x.mapExpression(l) for x in self.items ])
    def walk(self):
        yield self
        for x in self.items:
            for y in x.walk():
                yield y
    def usedCoefficients(self):
        xs = []
        ys = []
        for child in self.walk():
            if isinstance(child,Primitive):
                if child.k == 'circle':
                    xs.append(child.arguments[0].m)
                    ys.append(child.arguments[1].m)
                if child.k == 'rectangle' or child.k == 'line':
                    xs.append(child.arguments[0].m)
                    ys.append(child.arguments[1].m)
                    xs.append(child.arguments[2].m)
                    ys.append(child.arguments[3].m)
        return set([c for c in xs if c != 0 ]),set([c for c in ys if c != 0 ])
    def usedReflections(self):
        xs = []
        ys = []
        for child in self.walk():
            if isinstance(child,Reflection):
                if 'x' == child.axis:
                    xs.append(child.coordinate)
                else:
                    ys.append(child.coordinate)
        return set(xs),set(ys)

    def cost(self): return sum([x.cost() for x in self.items ])

    def totalCost(self):
        # cost taking into account the used coefficients & used boundaries
        c = self.cost()
        xs,ys = self.usedCoefficients()
        boundaryCount = 0
        for x in self.walk():
            if isinstance(x,Loop) and x.boundary != None: boundaryCount += 1
        boundaryCount = 0
        return 3*c + max(len(xs) - 1, 0) + max(len(ys) - 1, 0) + boundaryCount

    def optimizeUsingRewrites(self,d = 4):
        candidates = self.rewriteUpToDepth(d)
        scoredCandidates = [ (c.totalCost(),c) for c in candidates ]
        return min(scoredCandidates)

    def abstract(self,other,e):
        assert isinstance(other,Block)
        # We need to try removing stuff from the blocks to make them the same length.
        # For each way of removing stuff to make than the same length,
        # we have to consider every permutation of the block elements.
        # This is only efficient as long as the block bodies are small,
        # which holds in practice.
        for l in range(min(len(self.items),len(other.items)),0,-1):
            for p in iterationTools.combinations(self.items,l):
                for q in iterationTools.permutations(other.items,l):
                    try:
                        e_ = e
                        items = []
                        for x,y in zip(p,q):
                            a,e_ = x.abstract(y,e_)
                            items.append(a)
                        return Block(items),e_
                    except AbstractionFailure: pass
        raise AbstractionFailure

    def substitute(self,e):
        return Block([x.substitute(e) for x in self.items ])

    def usedLoops(self):
        for x in self.walk():
            if isinstance(x,Loop):
                yield {'depth': {'i':0,'j':1}[x.v],
                       'coefficient': x.bound.m,
                       'variable': {'i':0,'j':1,None:None}[x.bound.x],
                       'intercept': x.bound.b}

    def depth(self):
        return max([x.depth() for x in self.items ])
            
                        
            

# return something that resembles a syntax tree, built using the above classes
def parseSketchOutput(output, environment = None, loopDepth = 0, coefficients = None):
    commands = []
    # variable bindings introduced by the sketch: we have to resolve them
    environment = {} if environment == None else environment

    # global coefficients for linear transformations
    coefficients = {} if coefficients == None else coefficients

    output = output.split('\n')

    def getBlock(name, startingIndex, startingDepth = 0):
        d = startingDepth

        while d > -1:
            if 'dummyStart' in output[startingIndex] and name in output[startingIndex]:
                d += 1
            elif 'dummyEnd' in output[startingIndex] and name in output[startingIndex]:
                d -= 1
            startingIndex += 1

        return startingIndex

    def getBoundary(startingIndex):
        while True:
            if 'dummyStartBoundary' in output[startingIndex]:
                return getBlock('Boundary', startingIndex + 1)
            if 'dummyStartLoop' in output[startingIndex]:
                return None
            if 'dummyEndLoop' in output[startingIndex]:
                return None
            startingIndex += 1
                

    j = 0
    while j < len(output):
        l = output[j]
        if 'void renderSpecification' in l: break

        m = re.search('validate[X|Y]\((.*), (.*)\);',l)
        if m:
            environment[m.group(2)] = m.group(1)
            j += 1
            continue

        m = re.search('int\[[0-9]\] coefficients([1|2]) = {([,0-9\-]+)};',l)
        if m:
            coefficients[int(m.group(1))] = map(int,m.group(2).split(","))
        
        # apply the environment
        for v in sorted(environment.keys(), key = lambda v: -len(v)):
            l = l.replace(v,environment[v])

        # Apply the coefficients
        if 'coefficients' in l:
            for k in coefficients:
                for coefficientIndex,coefficientValue in enumerate(coefficients[k]):
                    pattern = '\(coefficients%s[^\[]*\[%d\]\)'%(k,coefficientIndex)
                    # print "Substituting the following pattern",pattern
                    # print "For the following value",coefficientValue
                    lp = re.sub(pattern, str(coefficientValue), l)
                    # if l != lp:
                    #     print "changed it to",lp
                    l = lp
        
        pattern = '\(\(\(shapeIdentity == 0\) && \(cx.* == (.+)\)\) && \(cy.* == (.+)\)\)'
        m = re.search(pattern,l)
        if m:
            x = parseExpression(m.group(1))
            y = parseExpression(m.group(2))
            commands += [Primitive('circle',x,y)]
            j += 1
            continue

        pattern = 'shapeIdentity == 1\) && \((.*) == lx1.*\)\) && \((.*) == ly1.*\)\) && \((.*) == lx2.*\)\) && \((.*) == ly2.*\)\) && \(([01]) == dashed\)\) && \(([01]) == arrow'
        m = re.search(pattern,l)
        if m:
            if False:
                print "Reading line!"
                print l
                for index in range(5): print "index",index,"\t",m.group(index),'\t',parseExpression(m.group(index))
            commands += [Primitive('line',
                                   parseExpression(m.group(1)),
                                   parseExpression(m.group(2)),
                                   parseExpression(m.group(3)),
                                   parseExpression(m.group(4)),
                                   'arrow = %s'%(m.group(6) == '1'),
                                   'solid = %s'%(m.group(5) == '0'))]         
            j += 1
            continue
        

        pattern = '\(\(\(\(\(shapeIdentity == 2\) && \((.+) == rx1.*\)\) && \((.+) == ry1.*\)\) && \((.+) == rx2.*\)\) && \((.+) == ry2.*\)\)'
        m = re.search(pattern,l)
        if m:
            # print m,m.group(1),m.group(2),m.group(3),m.group(4)
            commands += [Primitive('rectangle',
                                   parseExpression(m.group(1)),
                                   parseExpression(m.group(2)),
                                   parseExpression(m.group(3)),
                                   parseExpression(m.group(4)))]
            j += 1
            continue

        pattern = 'for\(int (.*) = 0; .* < (.*); .* = .* \+ 1\)'
        m = re.search(pattern,l)
        if m and (not ('reflectionIndex' in m.group(1))):
            boundaryIndex = getBoundary(j + 1)
            if boundaryIndex != None:
                boundary = "\n".join(output[(j+1):boundaryIndex])
                boundary = parseSketchOutput(boundary, environment, loopDepth + 1, coefficients)
                j = boundaryIndex
            else:
                boundary = None
            
            bodyIndex = getBlock('Loop', j+1)
            body = "\n".join(output[(j+1):bodyIndex])
            j = bodyIndex

            bound = parseExpression(m.group(2))
            body = parseSketchOutput(body, environment, loopDepth + 1, coefficients)
            v = ['i','j'][loopDepth]
            if v == 'j' and boundary != None and False:
                print "INNERLOOP"
                print '\n'.join(output)
                print "ENDOFINNERLOOP"
            commands += [Loop(v, bound, body, boundary)]
            continue

        pattern = 'dummyStartReflection\(([0-9]+), ([0-9]+)\)'
        m = re.search(pattern,l)
        if m:
            bodyIndex = getBlock('Reflection', j+1)
            body = "\n".join(output[(j+1):bodyIndex])
            j = bodyIndex
            x = int(m.group(1))
            y = int(m.group(2))
            axis = 'x' if y == 0 else 'y'
            coordinate = max([x,y])
            commands += [Reflection(axis, coordinate,
                                    parseSketchOutput(body, environment, loopDepth, coefficients))]

        j += 1
            
        
    return Block(commands)

def parseExpression(e):
    #print "parsing expression",e
    try: return LinearExpression(0,None,int(e))
    except:
        factor = re.search('([\-0-9]+) \* ',e)
        if factor != None: factor = int(factor.group(1))
        else: # try negative number
            factor = re.search('\(-\(([0-9]+)\)\) \* ',e)
            if factor != None: factor = -int(factor.group(1))
        offset = re.search(' \+ ([\-0-9]+)',e)
        if offset != None: offset = int(offset.group(1))
        variable = re.search('\[(\d)\]',e)
        if variable != None: variable = ['i','j'][int(variable.group(1))]

        if factor == None:
            factor = 1
        if offset == None: offset = 0
        if variable == None:
            print e
            assert False
        #print "Parsed into:",LinearExpression(factor,variable,offset)
        return LinearExpression(factor,variable,offset)


def renderEvaluation(s, exportTo = None):
    parse = evaluate(eval(s))
    x0 = min([x for l in parse.lines for x in l.usedXCoordinates()  ])
    y0 = min([y for l in parse.lines for y in l.usedYCoordinates()  ])
    x1 = max([x for l in parse.lines for x in l.usedXCoordinates()  ])
    y1 = max([y for l in parse.lines for y in l.usedYCoordinates()  ])

    render([parse.TikZ()],showImage = exportTo == None,exportTo = exportTo,canvas = (x1+1,y1+1), x0y0 = (x0 - 1,y0 - 1))









icingModelOutput = '''void render (int shapeIdentity, int cx, int cy, int lx1, int ly1, int lx2, int ly2, bit dashed, bit arrow, int rx1, int ry1, int rx2, int ry2, ref bit _out)  implements renderSpecification/*tmpzqJj8W.sk:209*/
{
  _out = 0;
  assume (((shapeIdentity == 0) || (shapeIdentity == 1)) || (shapeIdentity == 2)): "Assume at tmpzqJj8W.sk:210"; //Assume at tmpzqJj8W.sk:210
  assume (shapeIdentity != 2): "Assume at tmpzqJj8W.sk:212"; //Assume at tmpzqJj8W.sk:212
  assume (!(dashed)): "Assume at tmpzqJj8W.sk:216"; //Assume at tmpzqJj8W.sk:216
  assume (!(arrow)): "Assume at tmpzqJj8W.sk:217"; //Assume at tmpzqJj8W.sk:217
  int[2] coefficients1 = {-3,28};
  int[2] coefficients2 = {-3,24};
  int[0] environment = {};
  int[1] coefficients1_0 = coefficients1[0::1];
  int[1] coefficients2_0 = coefficients2[0::1];
  dummyStartLoop();
  int loop_body_cost = 0;
  bit _pac_sc_s15_s17 = 0;
  for(int j = 0; j < 3; j = j + 1)/*Canonical*/
  {
    assert (j < 4); //Assert at tmpzqJj8W.sk:96 (1334757887901394789)
    bit _pac_sc_s31 = _pac_sc_s15_s17;
    if(!(_pac_sc_s15_s17))/*tmpzqJj8W.sk:103*/
    {
      int[1] _pac_sc_s31_s33 = {0};
      push(0, environment, j, _pac_sc_s31_s33);
      dummyStartLoop();
      int loop_body_cost_0 = 0;
      int boundary_cost = 0;
      bit _pac_sc_s15_s17_0 = 0;
      for(int j_0 = 0; j_0 < 3; j_0 = j_0 + 1)/*Canonical*/
      {
        assert (j_0 < 4); //Assert at tmpzqJj8W.sk:96 (-4325113148049933570)
        if(((j_0 > 0) && 1) && 1)/*tmpzqJj8W.sk:97*/
        {
          dummyStartBoundary();
          bit _pac_sc_s26 = _pac_sc_s15_s17_0;
          if(!(_pac_sc_s15_s17_0))/*tmpzqJj8W.sk:99*/
          {
            int[2] _pac_sc_s26_s28 = {0,0};
            push(1, _pac_sc_s31_s33, j_0, _pac_sc_s26_s28);
            int x_s39 = 0;
            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[1])) + 8, x_s39);
            int y_s43 = 0;
            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[0])) + 7, y_s43);
            int x2_s47 = 0;
            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[1])) + 9, x2_s47);
            int y2_s51 = 0;
            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[0])) + 7, y2_s51);
            assert ((x_s39 == x2_s47) || (y_s43 == y2_s51)); //Assert at tmpzqJj8W.sk:137 (2109344902378156491)
            bit _pac_sc_s26_s30 = 0 || (((((((shapeIdentity == 1) && (x_s39 == lx1)) && (y_s43 == ly1)) && (x2_s47 == lx2)) && (y2_s51 == ly2)) && (0 == dashed)) && (0 == arrow));
            int x_s39_0 = 0;
            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[0])) + 7, x_s39_0);
            int y_s43_0 = 0;
            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[1])) + 8, y_s43_0);
            int x2_s47_0 = 0;
            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[0])) + 7, x2_s47_0);
            int y2_s51_0 = 0;
            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[1])) + 9, y2_s51_0);
            assert ((x_s39_0 == x2_s47_0) || (y_s43_0 == y2_s51_0)); //Assert at tmpzqJj8W.sk:137 (8471357942716875626)
            boundary_cost = 2;
            _pac_sc_s26_s30 = _pac_sc_s26_s30 || (((((((shapeIdentity == 1) && (x_s39_0 == lx1)) && (y_s43_0 == ly1)) && (x2_s47_0 == lx2)) && (y2_s51_0 == ly2)) && (0 == dashed)) && (0 == arrow));
            _pac_sc_s26 = _pac_sc_s26_s30;
          }
          _pac_sc_s15_s17_0 = _pac_sc_s26;
          dummyEndBoundary();
        }
        bit _pac_sc_s31_0 = _pac_sc_s15_s17_0;
        if(!(_pac_sc_s15_s17_0))/*tmpzqJj8W.sk:103*/
        {
          int[2] _pac_sc_s31_s33_0 = {0,0};
          push(1, _pac_sc_s31_s33, j_0, _pac_sc_s31_s33_0);
          int x_s39_1 = 0;
          validateX(((coefficients1_0[0]) * (_pac_sc_s31_s33_0[1])) + 7, x_s39_1);
          int y_s43_1 = 0;
          validateY(((coefficients2_0[0]) * (_pac_sc_s31_s33_0[0])) + 7, y_s43_1);
          loop_body_cost_0 = 1;
          _pac_sc_s31_0 = 0 || (((shapeIdentity == 0) && (cx == x_s39_1)) && (cy == y_s43_1));
        }
        _pac_sc_s15_s17_0 = _pac_sc_s31_0;
      }
      assert (loop_body_cost_0 != 0); //Assert at tmpzqJj8W.sk:105 (710966093749967188)
      dummyEndLoop();
      loop_body_cost = (loop_body_cost_0 + boundary_cost) + 1;
      _pac_sc_s31 = _pac_sc_s15_s17_0;
    }
    _pac_sc_s15_s17 = _pac_sc_s31;
  }
  assert (loop_body_cost != 0); //Assert at tmpzqJj8W.sk:105 (-6090248756724217227)
  dummyEndLoop();
  _out = _pac_sc_s15_s17;
  minimize(3 * (loop_body_cost + 1))'''

icingLines = '''
void render (int shapeIdentity, int cx, int cy, int lx1, int ly1, int lx2, int ly2, bit dashed, bit arrow, int rx1, int ry1, int rx2, int ry2, ref bit _out)  implements renderSpecification/*tmpuV7thE.sk:217*/
{
  _out = 0;
  assume (((shapeIdentity == 0) || (shapeIdentity == 1)) || (shapeIdentity == 2)): "Assume at tmpuV7thE.sk:218"; //Assume at tmpuV7thE.sk:218
  assume (shapeIdentity != 0): "Assume at tmpuV7thE.sk:219"; //Assume at tmpuV7thE.sk:219
  assume (shapeIdentity != 2): "Assume at tmpuV7thE.sk:220"; //Assume at tmpuV7thE.sk:220
  assume (!(dashed)): "Assume at tmpuV7thE.sk:224"; //Assume at tmpuV7thE.sk:224
  assume (!(arrow)): "Assume at tmpuV7thE.sk:225"; //Assume at tmpuV7thE.sk:225
  int[0] environment = {};
  dummyStartLoop();
  int loop_body_cost = 0;
  bit _pac_sc_s17_s19 = 0;
  for(int j = 0; j < 3; j = j + 1)/*Canonical*/
  {
    assert (j < 4); //Assert at tmpuV7thE.sk:104 (38)
    bit _pac_sc_s33 = _pac_sc_s17_s19;
    if(!(_pac_sc_s17_s19))/*tmpuV7thE.sk:111*/
    {
      int[1] _pac_sc_s33_s35 = {0};
      push(0, environment, j, _pac_sc_s33_s35);
      dummyStartLoop();
      int loop_body_cost_0 = 0;
      bit _pac_sc_s17_s19_0 = 0;
      for(int j_0 = 0; j_0 < 2; j_0 = j_0 + 1)/*Canonical*/
      {
        assert (j_0 < 4); //Assert at tmpuV7thE.sk:104 (46)
        bit _pac_sc_s33_0 = _pac_sc_s17_s19_0;
        if(!(_pac_sc_s17_s19_0))/*tmpuV7thE.sk:111*/
        {
          int[2] _pac_sc_s33_s35_0 = {0,0};
          push(1, _pac_sc_s33_s35, j_0, _pac_sc_s33_s35_0);
          int x_s41 = 0;
          validateX(((-(3)) * (_pac_sc_s33_s35_0[1])) + 5, x_s41);
          int y_s45 = 0;
          validateY((3 * (_pac_sc_s33_s35_0[0])) + 1, y_s45);
          int x2_s49 = 0;
          validateX(((-(3)) * (_pac_sc_s33_s35_0[1])) + 6, x2_s49);
          int y2_s53 = 0;
          validateY((3 * (_pac_sc_s33_s35_0[0])) + 1, y2_s53);
          assert ((x_s41 == x2_s49) || (y_s45 == y2_s53)); //Assert at tmpuV7thE.sk:145 (234)
          bit _pac_sc_s33_s37 = 0 || (((((((shapeIdentity == 1) && (x_s41 == lx1)) && (y_s45 == ly1)) && (x2_s49 == lx2)) && (y2_s53 == ly2)) && (0 == dashed)) && (0 == arrow));
          int x_s41_0 = 0;
          validateX(((-(3)) * (_pac_sc_s33_s35_0[0])) + 7, x_s41_0);
          int y_s45_0 = 0;
          validateY((3 * (_pac_sc_s33_s35_0[1])) + 2, y_s45_0);
          int x2_s49_0 = 0;
          validateX(((-(3)) * (_pac_sc_s33_s35_0[0])) + 7, x2_s49_0);
          int y2_s53_0 = 0;
          validateY((3 * (_pac_sc_s33_s35_0[1])) + 3, y2_s53_0);
          assert ((x_s41_0 == x2_s49_0) || (y_s45_0 == y2_s53_0)); //Assert at tmpuV7thE.sk:145 (236)
          loop_body_cost_0 = 2;
          _pac_sc_s33_s37 = _pac_sc_s33_s37 || (((((((shapeIdentity == 1) && (x_s41_0 == lx1)) && (y_s45_0 == ly1)) && (x2_s49_0 == lx2)) && (y2_s53_0 == ly2)) && (0 == dashed)) && (0 == arrow));
          _pac_sc_s33_0 = _pac_sc_s33_s37;
        }
        _pac_sc_s17_s19_0 = _pac_sc_s33_0;
      }
      assert (loop_body_cost_0 != 0); //Assert at tmpuV7thE.sk:113 (30)
      dummyEndLoop();
      loop_body_cost = loop_body_cost_0 + 1;
      _pac_sc_s33 = _pac_sc_s17_s19_0;
    }
    _pac_sc_s17_s19 = _pac_sc_s33;
  }
  assert (loop_body_cost != 0); //Assert at tmpuV7thE.sk:113 (35)
  dummyEndLoop();
  _out = _pac_sc_s17_s19;
  minimize(3 * (loop_body_cost + 1))
'''
icingCircles = '''
void render (int shapeIdentity, int cx, int cy, int lx1, int ly1, int lx2, int ly2, bit dashed, bit arrow, int rx1, int ry1, int rx2, int ry2, ref bit _out)  implements renderSpecification/*tmpuqIHtn.sk:209*/
{
  _out = 0;
  assume (((shapeIdentity == 0) || (shapeIdentity == 1)) || (shapeIdentity == 2)): "Assume at tmpuqIHtn.sk:210"; //Assume at tmpuqIHtn.sk:210
  assume (shapeIdentity != 2): "Assume at tmpuqIHtn.sk:212"; //Assume at tmpuqIHtn.sk:212
  assume (shapeIdentity != 1): "Assume at tmpuqIHtn.sk:213"; //Assume at tmpuqIHtn.sk:213
  int[2] coefficients1 = {-3,24};
  int[2] coefficients2 = {-3,16};
  int[0] environment = {};
  int[1] coefficients1_0 = coefficients1[0::1];
  int[1] coefficients2_0 = coefficients2[0::1];
  dummyStartLoop();
  int loop_body_cost = 0;
  bit _pac_sc_s15_s17 = 0;
  for(int j = 0; j < 3; j = j + 1)/*Canonical*/
  {
    assert (j < 4); //Assert at tmpuqIHtn.sk:96 (38)
    bit _pac_sc_s31 = _pac_sc_s15_s17;
    if(!(_pac_sc_s15_s17))/*tmpuqIHtn.sk:103*/
    {
      int[1] _pac_sc_s31_s33 = {0};
      push(0, environment, j, _pac_sc_s31_s33);
      dummyStartLoop();
      int loop_body_cost_0 = 0;
      bit _pac_sc_s15_s17_0 = 0;
      for(int j_0 = 0; j_0 < 3; j_0 = j_0 + 1)/*Canonical*/
      {
        assert (j_0 < 4); //Assert at tmpuqIHtn.sk:96 (46)
        bit _pac_sc_s31_0 = _pac_sc_s15_s17_0;
        if(!(_pac_sc_s15_s17_0))/*tmpuqIHtn.sk:103*/
        {
          int[2] _pac_sc_s31_s33_0 = {0,0};
          push(1, _pac_sc_s31_s33, j_0, _pac_sc_s31_s33_0);
          int x_s39 = 0;
          validateX(((coefficients1_0[0]) * (_pac_sc_s31_s33_0[0])) + 7, x_s39);
          int y_s43 = 0;
          validateY(((coefficients2_0[0]) * (_pac_sc_s31_s33_0[1])) + 7, y_s43);
          loop_body_cost_0 = 1;
          _pac_sc_s31_0 = 0 || (((shapeIdentity == 0) && (cx == x_s39)) && (cy == y_s43));
        }
        _pac_sc_s15_s17_0 = _pac_sc_s31_0;
      }
      assert (loop_body_cost_0 != 0); //Assert at tmpuqIHtn.sk:105 (30)
      dummyEndLoop();
      loop_body_cost = loop_body_cost_0 + 1;
      _pac_sc_s31 = _pac_sc_s15_s17_0;
    }
    _pac_sc_s15_s17 = _pac_sc_s31;
  }
  assert (loop_body_cost != 0); //Assert at tmpuqIHtn.sk:105 (35)
  dummyEndLoop();
  _out = _pac_sc_s15_s17;
  minimize(3 * (loop_body_cost + 1))
'''

if __name__ == '__main__':
    p2 = parseSketchOutput(icingLines)
    print p2.pretty()
    assert False
    p1 = parseSketchOutput(icingCircles)
    p3 = Block(p1.items + p2.items)
    print p3
    for r in p3.rewrites():
        print r.pretty()
        print "CHILDREN:"
        for c in r.rewrites():
            print c.pretty()
            print c
            print c.convertToPython()
            showImage(c.convertToSequence().draw())
        print "ENDOFCHILDREN"
        
    assert False
    start = Block([Loop('j',LinearExpression(0,None,3),
                        Block([Primitive('x')]),
                        boundary = Block([Primitive('w', LinearExpression(5,'j',-1))])),
                   Loop('j',LinearExpression(0,None,2),
                        Block([Primitive('y',LinearExpression(9,'j',4))]))])
    for r in start.rewrites():
        print r
    e = parseSketchOutput(icingModelOutput)
#    e = [circle(4,10)] + [ _i for i in range(0,3) for _i in ([line(3*i + 1,4,3*i + 1,2,arrow = True,solid = True)] + reflect(y = 6)([circle(3*i + 1,1)] + [line(4,9,3*i + 1,6,arrow = True,solid = True)])) ]
    print e.pretty()
    for h in e.hoistReflection():
        print h
        showImage(fastRender(h.convertToSequence()))
#    print len(e)
    # print e
    # for p in e.extrapolations():
    #     showImage(fastRender(p.convertToSequence()))
