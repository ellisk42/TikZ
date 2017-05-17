from render import render
from random import random,choice
import numpy as np
from utilities import linesIntersect,truncatedNormal
import math

'''
Programs: evaluator maps environment to (trace, environment)
Expressions: evaluator maps environment to value
'''

MAXIMUMCOORDINATE = 16
RADIUSNOISE = 0.0
COORDINATENOISE = 0.0
NOISYXOFFSET = 0.0
NOISYYOFFSET = 0.0

def setRadiusNoise(n):
    global RADIUSNOISE
    RADIUSNOISE = n
def setCoordinateNoise(n):
    global COORDINATENOISE
    COORDINATENOISE = n

def sampleNoisyOffset():
    global NOISYXOFFSET
    global NOISYYOFFSET
    NOISYYOFFSET = truncatedNormal(-1,1)*0.2
    NOISYXOFFSET = truncatedNormal(-1,1)*0.2

def makeLabel(j): return 'L'+str(j)
LABELS = [ makeLabel(j) for j in range(4) ]

def randomCoordinate():
    return int(random()*(MAXIMUMCOORDINATE - 2)) + 1


def inbounds(p):
    if isinstance(p,tuple):
        return p[0] > 0 and p[0] < MAXIMUMCOORDINATE and p[1] > 0 and p[1] < MAXIMUMCOORDINATE
    return p >= 1 and p <= MAXIMUMCOORDINATE - 1


class Program():
    def TikZ(self):
        return "\n".join(self.evaluate([])[0])
    def noisyTikZ(self):
        return "\n".join(self.noisyEvaluate([])[0])
    def __eq__(self,o): return str(self) == str(o)
    def __ne__(self,o): return str(self) != str(o)

class Expression():
    pass

class Number(Expression):
    def __init__(self,n): self.n = n
    def __str__(self): return str(self.n)
    def __eq__(self,o):
        return isinstance(o,Number) and self.n == o.n
    def __ne__(self,o):
        return not (self == o)
    def __gt__(self,o): return self.n > o.n
    def __lt__(self,o): return self.n < o.n
    def evaluate(self, environment):
        return self.n
    def children(self): return []
    def substitute(self,old, new):
        if old == self: return new
        return self

class Variable(Expression):
    def __init__(self,j): self.j = j
    def __str__(self): return "V(%d)"%(self.j)
    def __eq__(self,o):
        return isinstance(o,Variable) and o.j == self.j
    def evaluate(self,e): return e[self.j]
    def children(self): return []
    def substitute(self,old, new):
        return self

class DefineConstant(Program):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return "local %s in"%(str(self.value))

    def evaluate(self,environment):
        return ([], [self.value.evaluate(environment)]+environment)

    def children(self): return self.value.children()
    def substitute(self,old, new):
        return DefineConstant(self.value.substitute(old, new))

class AbsolutePoint(Expression):
    def __init__(self,x,y):
        # if isinstance(x,int) or isinstance(x,float): x = Number(x)
        # if isinstance(y,int) or isinstance(y,float): y = Number(y)
        self.x = x
        self.y = y

    def translate(self,x,y):
        return AbsolutePoint(Number(self.x.n + x),
                             Number(self.y.n + y))

    def children(self): return [self.x,self.y]
    def substitute(self, old, new):
        return AbsolutePoint(self.x.substitute(old, new),
                             self.y.substitute(old, new))
    
    @staticmethod
    def sample():
        return AbsolutePoint(Number(randomCoordinate()), Number(randomCoordinate()))

    def __eq__(self,o):
        if not isinstance(o,AbsolutePoint): return False
        return self.x == o.x and self.y == o.y

    def __str__(self):
        return "(%s,%s)"%(str(self.x),str(self.y))

    def isValid(self,_): return True

    def evaluate(self, environment):
        return "(%s,%s)"%(str(self.x.evaluate(environment)),
                          str(self.y.evaluate(environment)))

    def noisyEvaluate(self, environment):
        y = self.y.evaluate(environment)
        x = self.x.evaluate(environment)
        x += truncatedNormal(-1,1)*COORDINATENOISE + NOISYXOFFSET
        y += truncatedNormal(-1,1)*COORDINATENOISE + NOISYYOFFSET
        return "(%.2f,%.2f)"%(x,y)
    
    def mutate(self):
        while True:
            if random() > 0.5:
                dx = choice([-2,-1,1,2])
                dy = 0
            else:
                dx = 0
                dy = choice([-2,-1,1,2])
            dp = (self.x.n + dx,self.y.n + dy)
            if inbounds(dp):
                return AbsolutePoint(Number(dp[0]),Number(dp[1]))

class RelativePoint():
    def __init__(self, parent, orientation):
        self.parent = parent
        self.orientation = orientation

    def children(self):
        return []

    def mutate(self):
        return RelativePoint(self.parent, choice(['east','west','north','south']))
    @staticmethod
    def sample():
        return RelativePoint(choice(LABELS),None).mutate()
    def __str__(self):
        return "(%s.%s)"%(self.parent,self.orientation)

    def isValid(self,parents):
        return self.parent in parents

    def evaluate(self, environment): return str(self)

def samplePoint():
#    if random() > 0.5: return RelativePoint.sample()
    return AbsolutePoint.sample()


class Line(Program):
    def __init__(self, points, arrow = False, solid = True):
        self.points = points
        self.arrow = arrow
        self.solid = solid

        if self.length() == 0.0:
#            raise Exception('Attempt to create line with zero length')
            pass

    def translate(self,x,y):
        return Line([p.translate(x,y) for p in self.points ],self.arrow, self.solid)
    def logPrior(self): return -math.log(14*14*14*14*2*2)

    def isDiagonal(self):
        return not (len(set(self.usedXCoordinates())) == 1 or len(set(self.usedYCoordinates())) == 1)
    
    def __sub__(self,o):
        if not isinstance(o,Line): return float('inf')
        dx = sum([ abs(x1 - x2) for x1,x2 in zip(o.usedXCoordinates(),self.usedXCoordinates()) ])
        dy = sum([ abs(x1 - x2) for x1,x2 in zip(o.usedYCoordinates(),self.usedYCoordinates()) ])
        return dx + dy

    def children(self): return self.points
    def substitute(self, old, new):
        return Line([ p.substitute(old, new) for p in self.points], self.arrow, self.solid)
    def intersects(self,o):
        if isinstance(o,Circle): return o.intersects(self)
        if isinstance(o,Rectangle): return o.intersects(self)
        if isinstance(o,Line):
            s = self
            # if they have different orientations and then do a small shrink
            if len(set(self.usedXCoordinates())) != len(set(o.usedXCoordinates())) or len(set(self.usedYCoordinates())) != len(set(o.usedYCoordinates())):
                o = o.epsilonShrink()
                s = self.epsilonShrink()
            return linesIntersect(AbsolutePoint(s.points[0].x.n,s.points[0].y.n),
                                  AbsolutePoint(s.points[1].x.n,s.points[1].y.n),
                                  AbsolutePoint(o.points[0].x.n,o.points[0].y.n),
                                  AbsolutePoint(o.points[1].x.n,o.points[1].y.n))

    def usedXCoordinates(self): return [p.x.n for p in self.points ]
    def usedYCoordinates(self): return [p.y.n for p in self.points ]
    
    def __str__(self):
        return "Line(%s, arrow = %s, solid = %s)"%(", ".join(map(str,self.points)), str(self.arrow), str(self.solid))

    @staticmethod
    def lineCommand(points, arrow, solid, noisy = False):
        if noisy:
            attributes = ["line width = %.2fcm"%(0.1 + truncatedNormal(-1,1)*0.04)]
        else:
            attributes = ["line width = 0.1cm"]
        if arrow:
            scale = 1.5
            if noisy: scale = 1.0 + random()
            attributes += ["-{>[scale = %f]}"%(round(scale,1))]
        if not solid: attributes += ["dashed"]
        if noisy: attributes += ["pencildraw"]
        a = ",".join(attributes)
        return "\\draw [%s] %s;" % (a," -- ".join(points))
    
    def mutate(self):
        a = self.arrow
        s = self.solid
        if random() < 0.2: a = not a
        if random() < 0.2: s = not s
        
        r = choice(self.points)
        ps = [ (p.mutate() if p == r else p) for p in self.points ]
        if not a: ps = sorted(ps,key = lambda p: (p.x.n,p.y.n))
        return Line(ps, arrow = a, solid = s)
    @staticmethod
    def sample():
        while True:
            a = random() > 0.5
            ps = [samplePoint(),samplePoint()]
            if not a: ps = sorted(ps,key = lambda p: (p.x.n,p.y.n))
            l = Line(ps, solid = random() > 0.5, arrow = a)
            if l.length() > 0: return l
        
    def isValid(self, parents):
        return all([p.isValid(parents) for p in self.points ])

    def evaluate(self, environment):
        return ([Line.lineCommand([ p.evaluate(environment) for p in self.points ],
                                  self.arrow,
                                  self.solid)],
                environment)

    def noisyEvaluate(self, environment):
        # short lines should have less noise added to their offsets
        if self.length() < 3:
            n = COORDINATENOISE
            setCoordinateNoise(n*self.length()/4.0*COORDINATENOISE)
        # 60% of the noise is applied equally to each coordinate
        # 40% of the noise is per coordinate
        setCoordinateNoise(0.4*COORDINATENOISE)
        points = [ eval(p.noisyEvaluate(environment)) for p in self.points ]
        setCoordinateNoise(COORDINATENOISE/0.4)
        dx = truncatedNormal(-1,1)*COORDINATENOISE*0.6
        dy = truncatedNormal(-1,1)*COORDINATENOISE*0.6
        points = [ str((x + dx,y + dy)) for (x,y) in points ]
        e = ([Line.lineCommand(points,
                               self.arrow,
                               self.solid,
                               noisy = True)],
             environment)
        if self.length() < 3:
            setCoordinateNoise(n)
        return e

    @staticmethod
    def absolute(x1,y1,x2,y2, arrow = False, solid = True):
        return Line([AbsolutePoint(x1,y1),
                     AbsolutePoint(x2,y2)],
                    arrow = arrow,
                    solid = solid)
    @staticmethod
    def absoluteNumbered(x1,y1,x2,y2, arrow = False, solid = True):
        return Line([AbsolutePoint(Number(x1),Number(y1)),
                     AbsolutePoint(Number(x2),Number(y2))],
                    arrow = arrow,
                    solid = solid)

    def length(self):
        [p1,p2] = self.points
        return ((p1.x.n - p2.x.n)**2 + (p1.y.n - p2.y.n)**2)**(0.5)

    def epsilonShrink(self):
        l = self.length()
        if l < 0.001: return self
        e = 0.1/l
        [p1,p2] = self.points
        # points online: t*p1 + (1 - t)*p2
        x1 = (1 - e)*p1.x.n + e*p2.x.n
        y1 = (1 - e)*p1.y.n + e*p2.y.n
        x2 = (1 - e)*p2.x.n + e*p1.x.n
        y2 = (1 - e)*p2.y.n + e*p1.y.n

        return Line.absoluteNumbered(x1,y1,x2,y2)
        
        

class Rectangle(Program):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
    def logPrior(self): return -math.log(14*14*14*14)
    def translate(self,x,y):
        return Rectangle(self.p1.translate(x,y),
                         self.p2.translate(x,y))
    @staticmethod
    def absolute(x1,y1,x2,y2):
        return Rectangle(AbsolutePoint(Number(x1),Number(y1)),
                         AbsolutePoint(Number(x2),Number(y2)))
    def children(self): return [self.p1,self.p2]
    def substitute(self, old, new):
        return Rectangle(self.p1.substitute(old, new),self.p2.substitute(old, new))
    def constituentLines(self):
        return [Line([self.p1, AbsolutePoint(self.p2.x,self.p1.y)]),
                Line([AbsolutePoint(self.p2.x,self.p1.y), self.p2]),
                Line([self.p2, AbsolutePoint(self.p1.x,self.p2.y)]),
                Line([AbsolutePoint(self.p1.x,self.p2.y), self.p1])]
    def attachmentPoints(self):
        # all of the edges
        ps = [ (x, self.p1.y.n, 'v') for x in range(self.p1.x.n + 1, self.p2.x.n) ]
        ps += [ (self.p2.x.n, y, 'h') for y in range(self.p1.y.n + 1, self.p2.y.n) ]
        ps += [ (x, self.p2.y.n, 'v') for x in range(self.p1.x.n + 1, self.p2.x.n) ]
        ps += [ (self.p1.x.n, y, 'h') for y in range(self.p1.y.n + 1, self.p2.y.n) ]
        return ps
    def usedXCoordinates(self):
        return [self.p1.x.n,self.p2.x.n]
    def usedYCoordinates(self):
        return [self.p1.y.n,self.p2.y.n]
    
    def intersects(self,o):
        if isinstance(o,Circle): return o.intersects(self)
        if isinstance(o,Line):
            o = o.epsilonShrink() # lines are allowed to border rectangles
            for l in self.constituentLines():
                if l.intersects(o): return True
            return False
        if isinstance(o,Rectangle):
            for l1 in self.constituentLines():
                for l2 in o.constituentLines():
                    if l1.intersects(l2): return True
            return False
        raise Exception('rectangle intersection')
        
    
    @staticmethod
    def command(p1,p2, noisy = False):
        attributes = ["line width = 0.1cm"]
        if noisy:
            attributes = ["line width = %.2fcm"%(0.1 + truncatedNormal(-1,1)*0.04)]
        if noisy: attributes += ["pencildraw"]
        attributes = ",".join(attributes)
        (x1,y1) = eval(p1)
        (x2,y2) = eval(p2)
        p1 = "(%.2f,%.2f)"%(x1,y1)
        p2 = "(%.2f,%.2f)"%(x2,y1)
        p3 = "(%.2f,%.2f)"%(x2,y2)
        p4 = "(%.2f,%.2f)"%(x1,y2)
        return "\\draw [%s] %s -- %s -- %s -- %s -- cycle;"%(attributes,p1,p2,p3,p4)

    @staticmethod
    def noisyLineCommand(p1,p2,p3,p4, noisy = True):
        attributes = ["line width = 0.1cm"]
        if noisy:
            attributes = ["line width = %.2fcm"%(0.1 + truncatedNormal(-1,1)*0.04)]
        if noisy: attributes += ["pencildraw"]
        attributes = ",".join(attributes)
        return "\\draw [%s] %s -- %s -- %s -- %s -- cycle;"%(attributes,
                                                             p1,p2,p3,p4)

    def evaluate(self,environment):
        return ([Rectangle.command(self.p1.evaluate(environment),
                                   self.p2.evaluate(environment))],
                environment)
    def noisyEvaluate(self,environment):
        (x1,y1) = eval(self.p1.evaluate(environment))
        (x2,y2) = eval(self.p2.evaluate(environment))
        # perturb the center
        def centerNoise():
            return truncatedNormal(-1,1)*COORDINATENOISE*0.7
        def vertexNoise():
            return truncatedNormal(-1,1)*COORDINATENOISE*0.3
        w = x2 - x1
        h = y2 - y1
        cx = (x2 + x1)/2.0 + centerNoise()
        cy = (y2 + y1)/2.0 + centerNoise()
        x1 = cx - w/2.0 + vertexNoise() + NOISYXOFFSET
        x2 = cx + w/2.0 + vertexNoise() + NOISYXOFFSET
        y1 = cy - h/2.0 + vertexNoise() + NOISYYOFFSET
        y2 = cy + h/2.0 + vertexNoise() + NOISYYOFFSET
        
        p1 = "(%.2f,%.2f)"%(x1,y1)
        p2 = "(%.2f,%.2f)"%(x2,y1)
        p3 = "(%.2f,%.2f)"%(x2,y2)
        p4 = "(%.2f,%.2f)"%(x1,y2)
        return ([Rectangle.noisyLineCommand(p1,p2,p3,p4)],
                environment)
    def __str__(self):
        return "Rectangle(%s, %s)"%(str(self.p1),str(self.p2))
    def mutate(self):
        while True:
            p1 = self.p1
            p2 = self.p2
            if random() > 0.5:
                p1 = p1.mutate()
            else:
                p2 = p2.mutate()
            if p1.x.n < p2.x.n and p1.y.n < p2.y.n:
                return Rectangle(p1,p2)
        
    @staticmethod
    def sample():
        while True:
            p1 = AbsolutePoint.sample()
            p2 = AbsolutePoint.sample()
            if p1.x != p2.x and p1.y != p2.y:
                x1 = Number(min([p1.x.n,p2.x.n]))
                x2 = Number(max([p1.x.n,p2.x.n]))
                y1 = Number(min([p1.y.n,p2.y.n]))
                y2 = Number(max([p1.y.n,p2.y.n]))
                p1 = AbsolutePoint(x1,y1)
                p2 = AbsolutePoint(x2,y2)
                return Rectangle(p1, p2)

class Circle(Program):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def translate(self,x,y):
        return Circle(self.center.translate(x,y),
                      self.radius)

    @staticmethod
    def absolute(x,y): return Circle(AbsolutePoint(Number(x),Number(y)),Number(1))

    def logPrior(self): return -math.log(14*14)

    def children(self): return [self.center,self.radius]
    def substitute(self, old, new):
        return Circle(self.center.substitute(old, new),
                      self.radius)
    def attachmentPoints(self):
        r = self.radius.n
        x = self.center.x.n
        y = self.center.y.n
        return [(x + r,y,'h'),
                (x - r,y,'h'),
                (x,y + r,'v'),
                (x,y - r,'v')]
    def usedXCoordinates(self):
        return [self.center.x.n,
                self.center.x.n + self.radius.n,
                self.center.x.n - self.radius.n]
    def usedYCoordinates(self):
        return [self.center.y.n,
                self.center.y.n + self.radius.n,
                self.center.y.n - self.radius.n]
    
    @staticmethod
    def command(center, radius, noisy = False):
        noisy = "pencildraw," if noisy else ""
        radius = float(str(radius))
        lw = "line width = 0.1cm"
        if noisy:
            lw = "line width = %.2fcm"%(0.1 + truncatedNormal(-1,1)*0.03)
        return "\\node[draw,%scircle,inner sep=0pt,minimum size = %.2fcm,%s] at %s {};"%(noisy,radius*2,lw,center)
    def __str__(self):
        return "Circle(center = %s, radius = %s)"%(str(self.center),str(self.radius))
    def labeled(self,label):
        return "\\node(%s)[draw,circle,inner sep=0pt,minimum size = %dcm,line width = 0.1cm] at %s {};"%(label, self.radius*2, self.center)
    def mutate(self):
        while True:
            c = Circle(self.center.mutate(), self.radius)
            if c.inbounds():
                return c
    def intersects(self,o):
        if isinstance(o,Circle):
            x1,y1,r1 = self.center.x.n,self.center.y.n,self.radius.n
            x2,y2,r2 = o.center.x.n,o.center.y.n,o.radius.n
            return (x1 - x2)**2 + (y1 - y2)**2 < (r1 + r2)**2
        elif isinstance(o,Line):
            l = o
            c = self
            cx,cy = c.center.x.n,c.center.y.n
            r2 = c.radius.n*c.radius.n
            x2,y2 = l.points[1].x.n,l.points[1].y.n
            x1,y1 = l.points[0].x.n,l.points[0].y.n

            # I guess I should do the quadratic equation, but this is easier to code
            steps = 10
            for t in range(steps+1):
                t = float(t)/steps
                x = x1*t + x2*(1 - t)
                y = y1*t + y2*(1 - t)
                d2 = (x - cx)*(x - cx) + (y - cy)*(y - cy)
                if d2 < r2: return True
            return False
        elif isinstance(o,Rectangle):
            for l in o.constituentLines():
                if self.intersects(l): return True
            return False
            
    def inbounds(self):
        return inbounds(self.center.x.n + self.radius.n) and inbounds(self.center.x.n - self.radius.n) and inbounds(self.center.y.n + self.radius.n) and inbounds(self.center.y.n - self.radius.n)
    @staticmethod
    def sample():
        while True:
            p = AbsolutePoint.sample()
            r = Number(1)
            c = Circle(p,r)
            if c.inbounds():
                return c

    def evaluate(self, environment):
        return ([Circle.command(self.center.evaluate(environment),
                                self.radius.evaluate(environment))],
                environment)
    def noisyEvaluate(self, environment):
        r = self.radius.evaluate(environment) + truncatedNormal(-1,1)*RADIUSNOISE
        return ([Circle.command(self.center.noisyEvaluate(environment),
                                r,
                                noisy = True)],
                environment)

class Sequence(Program):
    def __init__(self, lines): self.lines = lines
    def __str__(self):
        lines = self.removeInvalidLines()
        notLines = [l for l in lines if not isinstance(l,Line) ]
        isLines = [l for l in lines if isinstance(l,Line) ]
        prefix = "\n".join([ str(l) for j,l in enumerate(notLines) ])
        suffix = "\n".join(map(str, isLines))
        return prefix + "\n" + suffix

    def logPrior(self):
        return sum([l.logPrior() for l in self.lines ]) - (len(self.lines) + 1)*math.log(4)

    def __eq__(self,o):
        if not isinstance(o,Sequence): return False
        return set(map(str,self.lines)) == set(map(str,o.lines))

    def removeDuplicates(self):
        return Sequence([ l for j,l in enumerate(self.lines) if not (str(l) in map(str,self.lines[:j])) ])

    def children(self): return self.lines
    def substitute(self, old, new):
        replacement = []
        for l in self.lines:
            replacement.append(l.substitute(old, new))
            # scoping indices
            if isinstance(l,DefineConstant) and isinstance(new, Variable):
                new = Variable(new.j + 1)
        return Sequence(replacement)
                

    def evaluate(self,environment):
        trace = []
        for p in self.lines:
            cs,e = p.evaluate(environment)
            trace += cs
            environment = e
        return (trace, environment)
    def noisyEvaluate(self,environment):
#        sampleNoisyOffset()
        trace = []
        for p in self.lines:
            cs,e = p.noisyEvaluate(environment)
            trace += cs
            environment = e
        return (trace, environment)
        
    @staticmethod
    def sample(sz = None):
        if sz == None:
            sz = choice([1,2,3])
        
        return Sequence([ Sequence.samplePart() for _ in range(sz) ])
    @staticmethod
    def samplePart():
        x = random()
        if x < 0.5: return Line.sample()
        return Circle.sample()

    def removeInvalidLines(self):
        notLines = [l for l in self.lines if not isinstance(l,Line) ]
        isLines = [l for l in self.lines if isinstance(l,Line) ]
        validParents = LABELS[:len(notLines)]
        validLines = [l for l in isLines if l.isValid(validParents) ]
        return validLines + notLines

    def hasCollisions(self):
        return any([ (j > k and l.intersects(lp))
                      for j,l in enumerate(self.lines)
                      for k,lp in enumerate(self.lines) ])

    def __hash__(self): return hash(str(self))
    def __len__(self): return len(self.lines)

    def mutate(self):
        r = random()
        if r < 0.3 or self.lines == []:
            n = randomLineOfCode()
            if n == None: n = []
            else: n = [n]
            return Sequence(self.lines + n)
        elif r < 0.6:
            r = choice(self.lines)
            return Sequence([ l for l in self.lines if l != r ])
        else:
            r = choice(self.lines)
            return Sequence([ (l if l != r else l.mutate()) for l in self.lines ])
    def framedRendering(self):
        parse = self
        x0 = min([x for l in parse.lines for x in l.usedXCoordinates()  ]) - 1
        y0 = min([y for l in parse.lines for y in l.usedYCoordinates()  ]) - 1
        x1 = max([x for l in parse.lines for x in l.usedXCoordinates()  ]) + 1
        y1 = max([y for l in parse.lines for y in l.usedYCoordinates()  ]) + 1

        x0 = min([x0,0])
        y0 = min([y0,0])
        x1 = max([x1,16])
        y1 = max([y1,16])
        return render([self.TikZ()],yieldsPixels = True,canvas = (x1,y1), x0y0 = (x0,y0))[0]

    def __sub__(self,o):
        return len(set(map(str,o.lines))^set(map(str,self.lines)))

    def translate(self,x,y):
        return Sequence([z.translate(x,y) for z in self.lines ])

    def canonicalTranslation(self):
        parse = self
        x0 = min([x for l in parse.lines for x in l.usedXCoordinates()  ])
        y0 = min([y for l in parse.lines for y in l.usedYCoordinates()  ])
        return self.translate(-x0,-y0)


def randomLineOfCode():
    k = choice(range(4))
    if k == 0: return None
    if k == 1: return Circle.sample()
    if k == 2: return Line.sample()
    if k == 3: return Rectangle.sample()
    assert False
    
if __name__ == '__main__':
    print Sequence([DefineConstant(Number(1)),
                    Circle(AbsolutePoint(Variable(0),Variable(0)), Number(1))]).TikZ()

    m = Line.absolute(Number(0),Number(0),
                      Number(5),Number(0))
    n = Line.absolute(Number(5),Number(0),
                      Number(5),Number(7))
    print m
    print n
    print m.intersects(n)
