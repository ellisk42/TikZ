from random import random,choice

MAXIMUMCOORDINATE = 8

def makeLabel(j): return 'L'+str(j)
LABELS = [ makeLabel(j) for j in range(4) ]

def randomCoordinate():
    return int(random()*(MAXIMUMCOORDINATE - 1)) + 1


def inbounds(p):
    if isinstance(p,tuple):
        return p[0] > 0 and p[0] < MAXIMUMCOORDINATE and p[1] > 0 and p[1] < MAXIMUMCOORDINATE
    return p >= 1 and p <= MAXIMUMCOORDINATE - 1

class AbsolutePoint():
    def __init__(self,x,y):
        self.x = x
        self.y = y
    @staticmethod
    def sample():
        return AbsolutePoint(randomCoordinate(), randomCoordinate())

    def __eq__(self,o):
        if not isinstance(o,AbsolutePoint): return False
        return self.x == o.x and self.y == o.y

    def __str__(self):
        return "(%d,%d)"%(self.x,self.y)

    def isValid(self,_): return True
    
    def mutate(self):
        while True:
            if random() > 0.5:
                dx = choice([-2,-1,1,2])
                dy = 0
            else:
                dx = 0
                dy = choice([-2,-1,1,2])
            dp = (self.x + dx,self.y + dy)
            if inbounds(dp):
                return AbsolutePoint(dp[0],dp[1])

class RelativePoint():
    def __init__(self, parent, orientation):
        self.parent = parent
        self.orientation = orientation

    def mutate(self):
        return RelativePoint(self.parent, choice(['east','west','north','south']))
    @staticmethod
    def sample():
        return RelativePoint(choice(LABELS),None).mutate()
    def __str__(self):
        return "(%s.%s)"%(self.parent,self.orientation)

    def isValid(self,parents):
        return self.parent in parents

def samplePoint():
    if random() > 0.5: return RelativePoint.sample()
    return AbsolutePoint.sample()


class Line():
    def __init__(self, points, arrow = False):
        self.points = points
        self.arrow = arrow
        
    def __str__(self):
        a = ",->" if self.arrow else ""
        return "\\draw [ultra thick%s] %s;" % (a," -- ".join([str(p) for p in self.points ]))
    
    def mutate(self):
        if random() < 0.2: return Line(self.points, not self.arrow)
        if random() < 0.2: return Line(list(reversed(self.points)), self.arrow)
        def mutatePoint(p):
            if isinstance(p, AbsolutePoint) and random() < 0.3: return RelativePoint.sample()
            return p.mutate()
        r = choice(self.points)
        return Line([ (mutatePoint(p) if p == r else p) for p in self.points ], self.arrow)
    @staticmethod
    def sample():
        return Line([samplePoint(),samplePoint()], random() > 0.5)
    def isValid(self, parents):
        return all([p.isValid(parents) for p in self.points ])

    @staticmethod
    def absolute(x1,y1,x2,y2):
        return Line([AbsolutePoint(x1,y1),
                     AbsolutePoint(x2,y2)])

class Rectangle():
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
    def __str__(self):
        return "\\draw [ultra thick] %s rectangle %s;" % (self.p1, self.p2)
    def mutate(self):
        if random() > 0.5:
            return Rectangle(randomPoint(),self.p2)
        else:
            return Rectangle(self.p1,randomPoint())
    @staticmethod
    def sample():
        while True:
            p1 = randomPoint()
            p2 = randomPoint()
            if p1[0] != p2[0] and p1[1] != p2[1]:
                return Rectangle(p1, p2)

class Circle():
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
    def __str__(self):
        return "\\node[draw,circle,inner sep=0pt,minimum size = %dcm,ultra thick] at %s {};"%(self.radius*2, self.center)
    def labeled(self,label):
        return "\\node(%s)[draw,circle,inner sep=0pt,minimum size = %dcm,ultra thick] at %s {};"%(label, self.radius*2, self.center)
    def mutate(self):
        while True:
            c = Circle(self.center.mutate(), self.radius)
            if c.inbounds():
                return c
    def inbounds(self):
        return inbounds(self.center.x + self.radius) and inbounds(self.center.x - self.radius) and inbounds(self.center.y + self.radius) and inbounds(self.center.y - self.radius)
    @staticmethod
    def sample():
        while True:
            p = AbsolutePoint.sample()
            r = 1
            c = Circle(p,r)
            if c.inbounds():
                return c

class Sequence():
    def __init__(self, lines): self.lines = lines
    def __str__(self):
        lines = self.removeInvalidLines()
        notLines = [l for l in lines if not isinstance(l,Line) ]
        isLines = [l for l in lines if isinstance(l,Line) ]
        prefix = "\n".join([ l.labeled(makeLabel(j)) for j,l in enumerate(notLines) ])
        suffix = "\n".join(map(str, isLines))
        return prefix + "\n" + suffix
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
