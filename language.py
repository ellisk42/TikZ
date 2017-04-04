from random import random,choice

'''
Programs: evaluator maps environment to (trace, environment)
Expressions: evaluator maps environment to value
'''

MAXIMUMCOORDINATE = 8

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

class Expression():
    pass

class Number(Expression):
    def __init__(self,n): self.n = n
    def __str__(self): return str(self.n)
    def __eq__(self,o):
        return isinstance(o,Number) and self.n == o.n
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
        
    def __str__(self):
        return Line.lineCommand(map(str,self.points), self.arrow, self.solid)

    @staticmethod
    def lineCommand(points, arrow, solid):
        attributes = ["ultra thick"]
        if arrow: attributes += ["->"]
        if not solid: attributes += ["dashed"]
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

    @staticmethod
    def absolute(x1,y1,x2,y2, arrow = False, solid = True):
        return Line([AbsolutePoint(x1,y1),
                     AbsolutePoint(x2,y2)],
                    arrow = arrow,
                    solid = solid)

# class Rectangle():
#     def __init__(self, p1, p2):
#         self.p1 = p1
#         self.p2 = p2
#     def __str__(self):
#         return "\\draw [ultra thick] %s rectangle %s;" % (self.p1, self.p2)
#     def mutate(self):
#         if random() > 0.5:
#             return Rectangle(randomPoint(),self.p2)
#         else:
#             return Rectangle(self.p1,randomPoint())
#     @staticmethod
#     def sample():
#         while True:
#             p1 = randomPoint()
#             p2 = randomPoint()
#             if p1[0] != p2[0] and p1[1] != p2[1]:
#                 return Rectangle(p1, p2)

class Circle():
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def children(self): return [self.center,self.radius]
    def substitute(self, old, new):
        return Circle(self.center.substitute(old, new),
                      self.radius)
    
    @staticmethod
    def command(center, radius):
        radius = int(str(radius))
        return "\\node[draw,circle,inner sep=0pt,minimum size = %dcm,ultra thick] at %s {};"%(radius*2, center)
    def __str__(self):
        return "Circle(center = %s, radius = %s)"%(str(self.center),str(self.radius))
    def labeled(self,label):
        return "\\node(%s)[draw,circle,inner sep=0pt,minimum size = %dcm,ultra thick] at %s {};"%(label, self.radius*2, self.center)
    def mutate(self):
        while True:
            c = Circle(self.center.mutate(), self.radius)
            if c.inbounds():
                return c
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
