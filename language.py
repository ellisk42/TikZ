from random import random,choice
import numpy as np
from utilities import linesIntersect,truncatedNormal

'''
Programs: evaluator maps environment to (trace, environment)
Expressions: evaluator maps environment to value
'''

MAXIMUMCOORDINATE = 8

RADIUSNOISE = 0.0
COORDINATENOISE = 0.0

def makeLabel(j): return 'L'+str(j)
LABELS = [ makeLabel(j) for j in range(4) ]

def randomCoordinate():
    return int(random()*(MAXIMUMCOORDINATE - 1)) + 1


def inbounds(p):
    if isinstance(p,tuple):
        return p[0] > 0 and p[0] < MAXIMUMCOORDINATE and p[1] > 0 and p[1] < MAXIMUMCOORDINATE
    return p >= 1 and p <= MAXIMUMCOORDINATE - 1


class Program():
    def TikZ(self):
        return "\n".join(self.evaluate([])[0])
    def noisyTikZ(self):
        return "\n".join(self.noisyEvaluate([])[0])

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
        self.x = x
        self.y = y

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
        x += truncatedNormal(-1,1)*COORDINATENOISE
        y += truncatedNormal(-1,1)*COORDINATENOISE
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

    def children(self): return self.points
    def substitute(self, old, new):
        return Line([ p.substitute(old, new) for p in self.points], self.arrow, self.solid)
    def intersects(self,o):
        if isinstance(o,Circle): return o.intersects(self)
        if isinstance(o,Rectangle): return o.intersects(self)
        if isinstance(o,Line):
            return linesIntersect(AbsolutePoint(self.points[0].x.n,self.points[0].y.n),
                                  AbsolutePoint(self.points[1].x.n,self.points[1].y.n),
                                  AbsolutePoint(o.points[0].x.n,o.points[0].y.n),
                                  AbsolutePoint(o.points[1].x.n,o.points[1].y.n))
        
    def __str__(self):
        return Line.lineCommand(map(str,self.points), self.arrow, self.solid)

    @staticmethod
    def lineCommand(points, arrow, solid, noisy = False):
        attributes = ["ultra thick"]
        if arrow: attributes += ["->"]
        if not solid: attributes += ["dashed"]
        if noisy: attributes += ["pencildraw"]
        a = ",".join(attributes)
        return "\\draw [%s] %s;" % (a," -- ".join(points))
    
    def mutate(self):
        if random() < 0.2: return Line(self.points, not self.arrow)
        if random() < 0.2: return Line(self.points, self.arrow,not self.solid)
        if random() < 0.2: return Line(list(reversed(self.points)), self.arrow,self.solid)
        def mutatePoint(p):
            return p.mutate()
        r = choice(self.points)
        return Line([ (mutatePoint(p) if p == r else p) for p in self.points ], self.arrow,self.solid)
    @staticmethod
    def sample():
        return Line([samplePoint(),samplePoint()], random() > 0.5, random() > 0.5)
    def isValid(self, parents):
        return all([p.isValid(parents) for p in self.points ])

    def evaluate(self, environment):
        return ([Line.lineCommand([ p.evaluate(environment) for p in self.points ],
                                  self.arrow,
                                  self.solid)],
                environment)

    def noisyEvaluate(self, environment):
        return ([Line.lineCommand([ p.noisyEvaluate(environment) for p in self.points ],
                                  self.arrow,
                                  self.solid,
                                  noisy = True)],
                environment)

    @staticmethod
    def absolute(x1,y1,x2,y2, arrow = False, solid = True):
        return Line([AbsolutePoint(x1,y1),
                     AbsolutePoint(x2,y2)],
                    arrow = arrow,
                    solid = solid)

class Rectangle():
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
    def children(self): return [self.p1,self.p2]
    def substitute(self, old, new):
        return Rectangle(self.p1.substitute(old, new),self.p2.substitute(old, new))
    def constituentLines(self):
        return [Line([self.p1, AbsolutePoint(self.p2.x,self.p1.y)]),
                Line([AbsolutePoint(self.p2.x,self.p1.y), self.p2]),
                Line([self.p2, AbsolutePoint(self.p1.x,self.p2.y)]),
                Line([AbsolutePoint(self.p1.x,self.p2.y), self.p1])]
    def intersects(self,o):
        if isinstance(o,Circle): return o.intersects(self)
        if isinstance(o,Line):
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
        attributes = ["ultra thick"]
        if noisy: attributes += ["pencildraw"]
        attributes = ",".join(attributes)
        return "\\draw [%s] %s rectangle %s;"%(attributes,p1,p2)

    def evaluate(self,environment):
        return ([Rectangle.command(self.p1.evaluate(environment),
                                   self.p2.evaluate(environment))],
                environment)
    def noisyEvaluate(self,environment):
        return ([Rectangle.command(self.p1.noisyEvaluate(environment),
                                   self.p2.noisyEvaluate(environment),
                                   noisy = True)],
                environment)
    def __str__(self):
        return "Rectangle(%s, %s)"%(str(self.p1),str(self.p2))
    def mutate(self):
        if random() > 0.5:
            return Rectangle(self.p1.mutate(),self.p2)
        else:
            return Rectangle(self.p1,self.p2.mutate())
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

class Circle():
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def children(self): return [self.center,self.radius]
    def substitute(self, old, new):
        return Circle(self.center.substitute(old, new),
                      self.radius)
    
    @staticmethod
    def command(center, radius, noisy = False):
        noisy = "pencildraw," if noisy else ""
        radius = float(str(radius))
        return "\\node[draw,%scircle,inner sep=0pt,minimum size = %.2fcm,ultra thick] at %s {};"%(noisy,radius*2, center)
    def __str__(self):
        return "Circle(center = %s, radius = %s)"%(str(self.center),str(self.radius))
    def labeled(self,label):
        return "\\node(%s)[draw,circle,inner sep=0pt,minimum size = %dcm,ultra thick] at %s {};"%(label, self.radius*2, self.center)
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

    def __eq__(self,o): return str(self) == str(o)
    def __ne__(self,o): return str(self) != str(o)
    def __hash__(self): return hash(str(self))
    def __len__(self): return len(self.lines)

    def mutate(self):
        r = random()
        if r < 0.3 or self.lines == []:
            return Sequence(self.lines + [Sequence.samplePart()])
        elif r < 0.6:
            r = choice(self.lines)
            return Sequence([ l for l in self.lines if l != r ])
        else:
            r = choice(self.lines)
            return Sequence([ (l if l != r else l.mutate()) for l in self.lines ])


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
