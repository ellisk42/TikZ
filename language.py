from render import render
from random import random,choice
import numpy as np
from utilities import linesIntersect,truncatedNormal,showImage


import math

import cairo
from time import time

'''
Programs: evaluator maps to trace
Expressions: evaluator maps to value
'''

MAXIMUMCOORDINATE = 16
RADIUSNOISE = 0.0
COORDINATENOISE = 0.0
STROKESIZE = 2

def setRadiusNoise(n):
    global RADIUSNOISE
    RADIUSNOISE = n
def setCoordinateNoise(n):
    global COORDINATENOISE
    COORDINATENOISE = n

    

def randomCoordinate():
    return int(random()*(MAXIMUMCOORDINATE - 2)) + 1


def inbounds(p):
    if isinstance(p,tuple):
        return p[0] > 0 and p[0] < MAXIMUMCOORDINATE and p[1] > 0 and p[1] < MAXIMUMCOORDINATE
    return p >= 1 and p <= MAXIMUMCOORDINATE - 1


class Program():
    def TikZ(self):
        return "\n".join(self.evaluate())
    def noisyTikZ(self):
        return "\n".join(self.noisyEvaluate())
    def __eq__(self,o): return str(self) == str(o)
    def __ne__(self,o): return str(self) != str(o)

class Expression():
    pass



class AbsolutePoint(Expression):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __add__(self,o):
        return AbsolutePoint(self.x + o.x,self.y + o.y)
    def __sub__(self,o):
        return AbsolutePoint(self.x - o.x,self.y - o.y)
    def __mul__(self,s):
        return AbsolutePoint(s*self.x,s*self.y)

    def normalized(self):
        l = (self.x*self.x + self.y*self.y)**0.5
        return AbsolutePoint(self.x/l,self.y/l)
    def rotateNinetyDegrees(self):
        return AbsolutePoint(self.y,
                             -self.x)

    def translate(self,x,y):
        return AbsolutePoint((self.x + x),
                             (self.y + y))

    def children(self): return [self.x,self.y]
    
    @staticmethod
    def sample():
        return AbsolutePoint((randomCoordinate()), (randomCoordinate()))

    def __eq__(self,o):
        if not isinstance(o,AbsolutePoint): return False
        return self.x == o.x and self.y == o.y

    def __str__(self):
        return "(%s,%s)"%(str(self.x),str(self.y))

    def isValid(self,_): return True

    def evaluate(self):
        return (self.x,self.y)

    def noisyEvaluate(self):
        return (self.x + truncatedNormal(-1,1)*COORDINATENOISE,
                self.y + truncatedNormal(-1,1)*COORDINATENOISE)
    
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
                return AbsolutePoint((dp[0]),(dp[1]))


class Label():
    allowedLabels = ['A','B','C','X','Y','Z']
    def __init__(self, p, c):
        self.p = p
        self.c = c

    def draw(self,context):
        context.set_source_rgb(256,256,256)
        context.select_font_face("Courier", cairo.FONT_SLANT_NORMAL, 
                                 cairo.FONT_WEIGHT_BOLD)
        context.set_font_size(12)
        (x, y, width, height, dx, dy) = context.text_extents(self.c)
        context.move_to(self.p.x*16 - width/2, self.p.y*16 - height/2)
        context.scale(1,-1)
        context.show_text(self.c)
        context.scale(1,-1)
        context.stroke()

    def translate(self,x,y):
        return Label(self.p.translate(x,y),self.c)
    def logPrior(self): return -math.log(26*2*14*14)
    def intersects(self,o):
        return Circle(self.p,1).intersects(o)
    def attachmentPoints(self):
        return []
    def usedXCoordinates(self): return [self.p.x]
    def usedYCoordinates(self): return [self.p.y]
    def __str__(self): return "Label(%s, \"%s\")"%(self.p,self.c)
    def mutate(self):
        if random() < 0.5:
            return Label(self.p.mutate(),self.c)
        else:
            return Label(self.p,choice([l for l in Label.allowedLabels if l != self.c ]))

    @staticmethod
    def sample(): return Label(AbsolutePoint.sample(),
                               choice(Label.allowedLabels))
                               #chr(ord(choice(['a','A'])) + choice(range(26))))
    def evaluate(self):
        return ["\\node at %s {\\Huge \\textbf{%s}};"%(self.p.evaluate(), self.c)]
    def noisyEvaluate(self):
        return ["\\node at %s {\\Huge \\textbf{%s}};"%(self.p.noisyEvaluate(), self.c)]
    
        


        
class Line(Program):
    def __init__(self, points, arrow = False, solid = True):
        self.points = points
        self.arrow = arrow
        self.solid = solid

        if self.length() == 0.0:
            # craise Exception('Attempt to create line with zero length')
            pass

    def draw(self,context):
        context.set_line_width(STROKESIZE)
        if not self.solid:
            context.set_dash([5,5])

        context.set_source_rgb(256,256,256)
        context.move_to(self.points[0].x*16,self.points[0].y*16)
        context.line_to(self.points[1].x*16,self.points[1].y*16)
        context.stroke()

        if not self.solid:
            context.set_dash([])

        if self.arrow:
            # corners of the arrow
            retreat = (self.points[0] - self.points[1]).normalized()*0.5 + self.points[1]
            wings = (self.points[1] - self.points[0]).rotateNinetyDegrees().normalized()*0.3
            k1 = retreat + wings
            k2 = retreat - wings
            
            context.move_to(self.points[1].x*16,self.points[1].y*16)
            for p in [k1,k2,self.points[1]]:
                context.line_to(p.x*16,p.y*16)
            context.fill()

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
    
    def intersects(self,o):
        if isinstance(o,Circle) or isinstance(o,Label) or isinstance(o,Rectangle):
            return o.intersects(self)
        if isinstance(o,Line):
            s = self
            # if they have different orientations and then do a small shrink
            if len(set(self.usedXCoordinates())) != len(set(o.usedXCoordinates())) or len(set(self.usedYCoordinates())) != len(set(o.usedYCoordinates())):
                o = o.epsilonShrink()
                s = self.epsilonShrink()
            return linesIntersect(AbsolutePoint(s.points[0].x,s.points[0].y),
                                  AbsolutePoint(s.points[1].x,s.points[1].y),
                                  AbsolutePoint(o.points[0].x,o.points[0].y),
                                  AbsolutePoint(o.points[1].x,o.points[1].y))

    def usedXCoordinates(self): return [p.x for p in self.points ]
    def usedYCoordinates(self): return [p.y for p in self.points ]
    
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
            if noisy: scale = 1.2 + random()*(1.5 - 1.2)*2
            scale = round(scale,1)
            differentStyles = ["-{>[scale = %f]}",
                               "-{Stealth[scale = %f]}",
                               "-{Latex[scale = %f]}"]
            if noisy: style = choice(differentStyles)
            else: style = differentStyles[0]
            attributes.append(style%(scale))
        if not solid:
            if not noisy: attributes += ["dashed"]
            else: attributes += ["dash pattern = on %dpt off %dpt"%(choice(range(5)) + 2,
                                                                    choice(range(5)) + 2)]
        if noisy: attributes += ["pencildraw"]
        a = ",".join(attributes)
        return "\\draw [%s] %s;" % (a," -- ".join(map(str,points)))
    
    def mutate(self):
        a = self.arrow
        s = self.solid
        ps = self.points
        
        mutateArrow = random() < 0.2
        mutateSolid = random() < 0.2

        if mutateSolid: s = not s
        if mutateArrow:
            if not a: # it didn't have an arrow and now it does
                # we need to randomly choose which side gets the arrow
                if random() < 0.5:
                    ps = list(reversed(ps))
            a = not a
        
        if random() < 0.4 or ((not mutateArrow) and (not mutateSolid)):
            r = choice(ps)
            ps = [ (p.mutate() if p == r else p) for p in ps ]
            
        if not a: ps = sorted(ps,key = lambda p: (p.x,p.y))
        mutant = Line(ps, arrow = a, solid = s)
        if mutant.length() < 1: return self.mutate()
        return mutant
    
    @staticmethod
    def sample():
        while True:
            a = random() > 0.5
            ps = [AbsolutePoint.sample(),AbsolutePoint.sample()]
            if not a: ps = sorted(ps,key = lambda p: (p.x,p.y))
            l = Line(ps, solid = random() > 0.5, arrow = a)
            if l.length() > 0: return l
        
    def evaluate(self):
        return [Line.lineCommand([ p.evaluate() for p in self.points ],
                                 self.arrow,
                                 self.solid)]

    def noisyEvaluate(self):
        # short lines should have less noise added to their offsets
        if self.length() < 3:
            n = COORDINATENOISE
            setCoordinateNoise(n*self.length()/4.0*COORDINATENOISE)
        # 60% of the noise is applied equally to each coordinate
        # 40% of the noise is per coordinate
        setCoordinateNoise(0.4*COORDINATENOISE)
        points = [ p.noisyEvaluate() for p in self.points ]
        setCoordinateNoise(COORDINATENOISE/0.4)
        dx = truncatedNormal(-1,1)*COORDINATENOISE*0.6
        dy = truncatedNormal(-1,1)*COORDINATENOISE*0.6
        points = [ str((x + dx,y + dy)) for (x,y) in points ]
        e = [Line.lineCommand(points,
                              self.arrow,
                              self.solid,
                              noisy = True)]
        if self.length() < 3:
            setCoordinateNoise(n)
        return e

    @staticmethod
    def absolute(x1,y1,x2,y2, arrow = False, solid = True):
        return Line([AbsolutePoint(x1,y1),
                     AbsolutePoint(x2,y2)],
                    arrow = arrow,
                    solid = solid)

    def length(self):
        [p1,p2] = self.points
        return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**(0.5)

    def epsilonShrink(self):
        l = self.length()
        if l < 0.001: return self
        e = 0.1/l
        [p1,p2] = self.points
        # points online: t*p1 + (1 - t)*p2
        x1 = (1 - e)*p1.x + e*p2.x
        y1 = (1 - e)*p1.y + e*p2.y
        x2 = (1 - e)*p2.x + e*p1.x
        y2 = (1 - e)*p2.y + e*p1.y

        return Line.absolute(x1,y1,x2,y2)
                

class Rectangle(Program):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        
    def draw(self,context):
        context.set_line_width(STROKESIZE)
        context.set_source_rgb(256,256,256)
        context.rectangle(self.p1.x*16,self.p1.y*16,
                          (self.p2.x - self.p1.x)*16,(self.p2.y - self.p1.y)*16)
        context.stroke()
    
    def logPrior(self): return -math.log(14*14*14*14)
    def translate(self,x,y):
        return Rectangle(self.p1.translate(x,y),
                         self.p2.translate(x,y))
    @staticmethod
    def absolute(x1,y1,x2,y2):
        return Rectangle(AbsolutePoint((x1),(y1)),
                         AbsolutePoint((x2),(y2)))
    def children(self): return [self.p1,self.p2]
    def constituentLines(self):
        return [Line([self.p1, AbsolutePoint(self.p2.x,self.p1.y)]),
                Line([AbsolutePoint(self.p2.x,self.p1.y), self.p2]),
                Line([self.p2, AbsolutePoint(self.p1.x,self.p2.y)]),
                Line([AbsolutePoint(self.p1.x,self.p2.y), self.p1])]
    def attachmentPoints(self):
        # all of the edges
        ps = [ (x, self.p1.y, 'v') for x in range(self.p1.x + 1, self.p2.x) ]
        ps += [ (self.p2.x, y, 'h') for y in range(self.p1.y + 1, self.p2.y) ]
        ps += [ (x, self.p2.y, 'v') for x in range(self.p1.x + 1, self.p2.x) ]
        ps += [ (self.p1.x, y, 'h') for y in range(self.p1.y + 1, self.p2.y) ]
        return ps
    def usedXCoordinates(self):
        return [self.p1.x,self.p2.x]
    def usedYCoordinates(self):
        return [self.p1.y,self.p2.y]
    
    def intersects(self,o):
        if isinstance(o,Circle) or isinstance(o,Label): return o.intersects(self)
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
        (x1,y1) = p1
        (x2,y2) = p2
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

    def evaluate(self):
        return [Rectangle.command(self.p1.evaluate(),
                                  self.p2.evaluate())]
    def noisyEvaluate(self):
        (x1,y1) = self.p1.evaluate()
        (x2,y2) = self.p2.evaluate()
        # perturb the center
        def centerNoise():
            return truncatedNormal(-1,1)*COORDINATENOISE*0.7
        def vertexNoise():
            return truncatedNormal(-1,1)*COORDINATENOISE*0.3
        w = x2 - x1
        h = y2 - y1
        cx = (x2 + x1)/2.0 + centerNoise()
        cy = (y2 + y1)/2.0 + centerNoise()
        x1 = cx - w/2.0 + vertexNoise()
        x2 = cx + w/2.0 + vertexNoise()
        y1 = cy - h/2.0 + vertexNoise()
        y2 = cy + h/2.0 + vertexNoise()
        
        p1 = "(%.2f,%.2f)"%(x1,y1)
        p2 = "(%.2f,%.2f)"%(x2,y1)
        p3 = "(%.2f,%.2f)"%(x2,y2)
        p4 = "(%.2f,%.2f)"%(x1,y2)
        return [Rectangle.noisyLineCommand(p1,p2,p3,p4)]
    def __str__(self):
        return "Rectangle(%s, %s)"%(str(self.p1),str(self.p2))
    def mutate(self):
        dx = self.p2.x - self.p1.x
        dy = self.p2.y - self.p1.y
        if dx == dy and dx < 8 and dx%2 == 0 and random() < 0.5:
            return Circle(AbsolutePoint(self.p1.x + dx/2,
                                        self.p1.y + dy/2),
                          dx/2)
        while True:
            p1 = self.p1
            p2 = self.p2
            if random() > 0.5:
                p1 = p1.mutate()
            else:
                p2 = p2.mutate()
            if p1.x < p2.x and p1.y < p2.y:
                return Rectangle(p1,p2)
        
    @staticmethod
    def sample():
        while True:
            p1 = AbsolutePoint.sample()
            p2 = AbsolutePoint.sample()
            if p1.x != p2.x and p1.y != p2.y:
                x1 = (min([p1.x,p2.x]))
                x2 = (max([p1.x,p2.x]))
                y1 = (min([p1.y,p2.y]))
                y2 = (max([p1.y,p2.y]))
                p1 = AbsolutePoint(x1,y1)
                p2 = AbsolutePoint(x2,y2)
                return Rectangle(p1, p2)

class Circle(Program):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def draw(self,context):
        context.set_line_width(STROKESIZE)
        context.set_source_rgb(256,256,256)
        context.arc(self.center.x*16,self.center.y*16,
                    self.radius*16,
                    0,
                    2*math.pi)
        context.stroke()

    def translate(self,x,y):
        return Circle(self.center.translate(x,y),
                      self.radius)

    @staticmethod
    def absolute(x,y): return Circle(AbsolutePoint((x),(y)),(1))

    def logPrior(self): return -math.log(14*14)

    def children(self): return [self.center,self.radius]
    def attachmentPoints(self):
        r = self.radius
        x = self.center.x
        y = self.center.y
        return [(x + r,y,'h'),
                (x - r,y,'h'),
                (x,y + r,'v'),
                (x,y - r,'v')]
    def usedXCoordinates(self):
        return [self.center.x,
                self.center.x + self.radius,
                self.center.x - self.radius]
    def usedYCoordinates(self):
        return [self.center.y,
                self.center.y + self.radius,
                self.center.y - self.radius]
    
    @staticmethod
    def command(center, radius, noisy = False):
        noisy = "pencildraw," if noisy else ""
        radius = float(radius)
        lw = "line width = 0.1cm"
        if noisy:
            lw = "line width = %.2fcm"%(0.1 + truncatedNormal(-1,1)*0.03)
        return "\\node[draw,%scircle,inner sep=0pt,minimum size = %.2fcm,%s] at %s {};"%(noisy,radius*2,lw,center)
    def __str__(self):
        return "Circle(center = %s, radius = %s)"%(str(self.center),str(self.radius))
    
    def mutate(self):
        if self.radius < 3 and random() < 0.15:
            return Rectangle.absolute(self.center.x - self.radius, self.center.y - self.radius,
                                      self.center.x + self.radius, self.center.y + self.radius)
        while True:
            if random() < 0.5:
                c = Circle(self.center.mutate(), self.radius)
            else:
                if self.radius < 2: r = self.radius + 1
                else: r = self.radius + choice([-1,1])
                c = Circle(self.center, r)
            if c.inbounds():
                return c
    def intersects(self,o):
        if isinstance(o,Label): return o.intersects(self)
        if isinstance(o,Circle):
            x1,y1,r1 = self.center.x,self.center.y,self.radius
            x2,y2,r2 = o.center.x,o.center.y,o.radius
            return (x1 - x2)**2 + (y1 - y2)**2 < (r1 + r2)**2
        elif isinstance(o,Line):
            l = o
            c = self
            cx,cy = c.center.x,c.center.y
            r2 = c.radius*c.radius
            x2,y2 = l.points[1].x,l.points[1].y
            x1,y1 = l.points[0].x,l.points[0].y

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
        return inbounds(self.center.x + self.radius) and inbounds(self.center.x - self.radius) and inbounds(self.center.y + self.radius) and inbounds(self.center.y - self.radius)
    @staticmethod
    def sample():
        while True:
            p = AbsolutePoint.sample()
            r = choice(range(5)) + 1
            c = Circle(p,r)
            if c.inbounds():
                return c

    def evaluate(self):
        return [Circle.command(self.center.evaluate(),
                               self.radius)]
    def noisyEvaluate(self):
        r = self.radius + truncatedNormal(-1,1)*RADIUSNOISE
        return [Circle.command(self.center.noisyEvaluate(),
                                r,
                                noisy = True)]

class Sequence(Program):
    def __init__(self, lines): self.lines = lines
    def __str__(self):
        return "\n".join(map(str,self.lines))

    def logPrior(self):
        return sum([l.logPrior() for l in self.lines ]) - (len(self.lines) + 1)*math.log(4)

    def __eq__(self,o):
        if not isinstance(o,Sequence): return False
        return set(map(str,self.lines)) == set(map(str,o.lines))

    def removeDuplicates(self):
        return Sequence([ l for j,l in enumerate(self.lines) if not (str(l) in map(str,self.lines[:j])) ])

    def children(self): return self.lines
    
    def evaluate(self):
        trace = []
        for p in self.lines:
            cs = p.evaluate()
            trace += cs
        return (trace)
    def noisyEvaluate(self):
        trace = []
        for p in self.lines:
            cs = p.noisyEvaluate()
            trace += cs
        return (trace)
        
    @staticmethod
    def sample(sz = None):
        if sz == None:
            sz = choice([1,2,3])
        
        return Sequence([ Sequence.samplePart() for _ in range(sz) ])
    @staticmethod
    def samplePart():
        while True:
            k = randomLineOfCode()
            if k != None: return k

    def hasCollisions(self):
        return any([ (j > k and l.intersects(lp))
                      for j,l in enumerate(self.lines)
                      for k,lp in enumerate(self.lines) ])

    def __hash__(self): return hash(str(self))
    def __len__(self): return len(self.lines)

    def mutate(self, canRemove = True):
        r = random()
        if r < 0.3 or self.lines == []:
            n = randomLineOfCode()
            if n == None: n = []
            else: n = [n]
            return Sequence(self.lines + n)
        elif r < 0.6 and canRemove:
            r = choice(self.lines)
            return Sequence([ l for l in self.lines if l != r ])
        else:
            r = choice(self.lines)
            return Sequence([ (l if l != r else l.mutate()) for l in self.lines ])
    def extent(self):
        parse = self
        x0 = min([x for l in parse.lines for x in l.usedXCoordinates()  ]) - 1
        y0 = min([y for l in parse.lines for y in l.usedYCoordinates()  ]) - 1
        x1 = max([x for l in parse.lines for x in l.usedXCoordinates()  ]) + 1
        y1 = max([y for l in parse.lines for y in l.usedYCoordinates()  ]) + 1

        x0 = min([x0,0])
        y0 = min([y0,0])
        x1 = max([x1,16])
        y1 = max([y1,16])

        return (x0,y0,x1,y1)
    def framedRendering(self, reference = None):
        (x0,y0,x1,y1) = self.extent()
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

    def draw(self,context = None):
        if context == None:
            data = np.zeros((256, 256), dtype=np.uint8)
            surface = cairo.ImageSurface.create_for_data(data,cairo.FORMAT_A8,256,256)
            context = cairo.Context(surface)
        for l in self.lines: l.draw(context)
        data = np.flip(data, 0)
        return data/255.0


def randomLineOfCode():
    k = choice(range(5))
    if k == 0: return None
    if k == 1: return Circle.sample()
    if k == 2: return Line.sample()
    if k == 3: return Rectangle.sample()
    if k == 4: return Label.sample()
    assert False
    
if __name__ == '__main__':
    
    s = Sequence.sample(10)
    print s
    x = render([s.noisyTikZ()],yieldsPixels = True)[0]
    y = (s.draw())

    showImage(np.concatenate([x,y]))

    # rendering benchmarking
    startTime = time()
    N = 100
    for _ in range(N):
        s.draw()
    print "%f fps"%(N/(time() - startTime))
