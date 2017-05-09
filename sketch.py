import re
import numpy as np
import cv2
from utilities import loadImage,showImage
from subprocess import Popen, PIPE
import os
import tempfile

from language import *

def synthesizeProgram(parse):
    parts = []
    hasCircles = False
    hasRectangles = False
    hasLines = False
    arrows = [] #}settings of the arrow parameters that are observed in the data
    solid = [] # settings of the solid parameters are so served in the data

    # translate drawing the lower left-hand corner
    x0 = min([x for l in parse.lines for x in l.usedXCoordinates()  ])
    y0 = min([y for l in parse.lines for y in l.usedYCoordinates()  ])
    x1 = max([x for l in parse.lines for x in l.usedXCoordinates()  ]) - x0
    y1 = max([y for l in parse.lines for y in l.usedYCoordinates()  ]) - y0
    biggestNumber = -1 #max([x1,y1,4])
    
    for p in parse.lines:
        if isinstance(p,Circle):
            parts.append("_c(%d,%d)"%(p.center.x.n - x0,
                                      p.center.y.n - y0))
            hasCircles = True
        elif isinstance(p,Rectangle):
            hasRectangles = True
            parts.append("_r(%d,%d,%d,%d)"%(p.p1.x.n - x0,
                                            p.p1.y.n - y0,
                                            p.p2.x.n - x0,
                                            p.p2.y.n - y0))
        elif isinstance(p,Line):
            hasLines = True
            arrows.append(p.arrow)
            solid.append(p.solid)
            parts.append("_l(%d,%d,%d,%d,%d,%d)"%(p.points[0].x.n - x0,
                                                  p.points[0].y.n - y0,
                                                  p.points[1].x.n - x0,
                                                  p.points[1].y.n - y0,
                                                  0 if p.solid else 1,
                                                  1 if p.arrow else 0))
    
    source = '''
pragma options "--bnd-unroll-amnt 4 --bnd-arr1d-size 2 --bnd-arr-size 2 --bnd-int-range %d";
    
#define MAXIMUMXCOORDINATE %d
#define MAXIMUMYCOORDINATE %d
#define HASCIRCLES %d
#define HASRECTANGLES %d
#define HASLINES %d
#define HASSOLID %d
#define HASDASHED %d
#define HASARROW %d
#define HASNOARROW %d

#include "common.skh"
bit renderSpecification(SHAPEVARIABLES) {
  assume shapeIdentity == CIRCLE || shapeIdentity == LINE || shapeIdentity == RECTANGLE;
  if (!HASCIRCLES) assume shapeIdentity != CIRCLE;
  if (!HASRECTANGLES) assume shapeIdentity != RECTANGLE;
  if (!HASLINES) assume shapeIdentity != LINE;
  else {
    if (!HASSOLID) assume dashed;
    if (!HASDASHED) assume !dashed;
    if (!HASARROW) assume !arrow;
    if (!HASNOARROW) assume arrow;
  }
  return %s;
}
'''%(biggestNumber,
     x1,y1,
     hasCircles,hasRectangles,hasLines,
     (True in solid),(False in solid),
     (True in arrows),(False in arrows),
     " || ".join(parts))

    fd = tempfile.NamedTemporaryFile(mode = 'w',suffix = '.sk',delete = False,dir = '.')
    fd.write(source)
    fd.close()

    # Temporary file for collecting the sketch output
    od = tempfile.NamedTemporaryFile(mode = 'w',delete = False,dir = '/tmp')
    od.write('') # just create the file that were going to overwrite
    od.close()
    outputFile = od.name

    os.system('sketch -V 10 %s 2> %s > %s'%(fd.name, outputFile, outputFile))

    output = open(outputFile,'r').read()
    os.remove(fd.name)
    os.remove(outputFile)

    # Recover the program length from the sketch output
    programSize = [ l for l in output.split('\n') if "*********INSIDE minimizeHoleValue" in l ] #if () {}
    if programSize == []:
        print output
        return None
    programSize = programSize[-1]
    m = re.match('.*=([0-9]+),',programSize)
    cost = int(m.group(1))

    # find the body of the synthesized code
    body = None
    for l in output.split('\n'):
        if body == None and 'void render ' in l:
            body = [l]
        elif body != None:
            body.append(l)
            if 'minimize' in l:
                break
    body = "\n".join(body)
    # print body
    # parseSketchOutput(body)
    # assert False
    return cost,body

def parseSketchOutput(output):
    commands = []
    environment = {} # variable bindings introduced by the sketch: we have to resolve them
    
    for l in output.split('\n'):
        if 'void renderSpecification' in l: break

        m = re.search('validate[X|Y]\((.*), (.*)\);',l)
        if m:
            environment[m.group(2)] = m.group(1)
            continue

        # apply the environment
        for v in sorted(environment.keys(), key = lambda v: -len(v)):
            # if v in l:
            #     print "Replacing %s w/ %s in %s gives %s"%(v,environment[v],l,l.replace(v,environment[v]))
            l = l.replace(v,environment[v])

        
        nestingDepth = 0
        while nestingDepth < len(l) and l[nestingDepth] == ' ': nestingDepth += 1
        #nest = ' '*nestingDepth
        
        pattern = '\(\(\(shapeIdentity == 0\) && \(cx.* == (.+)\)\) && \(cy.* == (.+)\)\)'
        m = re.findall(pattern,l)
        assert len(m) < 2
        if m != []:
            x = parseExpression(m[0][0])
            y = parseExpression(m[0][1])
            commands += [(nestingDepth, "Circle(%s,%s)"%(x,y))]
            continue

        pattern = 'shapeIdentity == 1\) && \((.*) == lx1.*\)\) && \((.*) == ly1.*\)\) && \((.*) == lx2.*\)\) && \((.*) == ly2.*\)\) && \(([01]) == dashed\)\) && \(([01]) == arrow'
        m = re.search(pattern,l)
        if m:
            commands += [(nestingDepth, "Line(%s,%s,%s,%s,arrow = %s,solid = %s)"%(parseExpression(m.group(1)),
                                                                                   parseExpression(m.group(2)),
                                                                                   parseExpression(m.group(3)),
                                                                                   parseExpression(m.group(4)),
                                                                                   m.group(6) == '1',
                                                                                   m.group(5) == '0'))]
            continue
        

        pattern = '\(\(\(\(\(shapeIdentity == 2\) && \((.+) == rx1.*\)\) && \((.+) == ry1.*\)\) && \((.+) == rx2.*\)\) && \((.+) == ry2.*\)\)'
        m = re.search(pattern,l)
        if m:
            # print m,m.group(1),m.group(2),m.group(3),m.group(4)
            commands += [(nestingDepth, "Rectangle(%s,%s,%s,%s)"%(parseExpression(m.group(1)),
                                            parseExpression(m.group(2)),
                                            parseExpression(m.group(3)),
                                                                  parseExpression(m.group(4))))]
            continue

        pattern = 'for\(int (.*) = 0; .* < (.*); .* = .* \+ 1\)'
        m = re.search(pattern,l)
        if m and (not ('reflectionIndex' in m.group(1))):
            commands += [(nestingDepth, "for (%s)"%(parseExpression(m.group(2))))]
            continue

        pattern = 'cx_.* = (\d+) - \(cx'
        m = re.search(pattern,l)
        if m:
            commands += [(nestingDepth, "reflect(x = %d)"%(int(m.group(1))))]
            continue
        pattern = 'cy_.* = (\d+) - \(cy'
        m = re.search(pattern,l)
        if m:
            commands += [(nestingDepth, "reflect(y = %d)"%(int(m.group(1))))]
            continue

        pattern = 'ry1.* = \((\d+) - \(ry1.* - rectangleHeight;'
        m = re.search(pattern,l)
        if m:
            commands += [(nestingDepth, "reflect(y = %d)"%(int(m.group(1))))]
            continue
        pattern = 'rx1.* = \((\d+) - \(rx1.* - rectangleWidth'
        m = re.search(pattern,l)
        if m:
            commands += [(nestingDepth, "reflect(x = %d)"%(int(m.group(1))))]
            continue

        #lx1_0 = 7 - (lx1_0 - 0);
        pattern = 'ly1.* = (\d+) - \(ly1.* - 0;'
        m = re.search(pattern,l)
        if m:
            commands += [(nestingDepth, "reflect(y = %d)"%(int(m.group(1))))]
            continue
        pattern = 'lx1.* = (\d+) - \(lx1.* - 0;'
        m = re.search(pattern,l)
        if m:
            commands += [(nestingDepth, "reflect(x = %d)"%(int(m.group(1))))]
            continue

    # we can pick up redundant reflections. remove them.
    noStutter = []
    for j,(d,c) in enumerate(commands):
        if j > 0 and 'reflect' in c and c == commands[j - 1][1]:
            continue
        noStutter.append((d,c))
    commands = noStutter
        
    return "\n".join([ " "*d+c for d,c in commands ])

def parseExpression(e):
    try: return int(e)
    except:
        factor = re.search('([\-0-9]+) * ',e)
        if factor != None: factor = int(factor.group(1))
        offset = re.search(' \+ ([\-0-9]+)',e)
        if offset != None: offset = int(offset.group(1))
        variable = re.search('\[(\d)\]',e)
        if variable != None: variable = ['i','j'][int(variable.group(1))]

        if factor == None: factor = 0
        if offset == None: offset = 0
        if variable == None:
            print e
            assert False

        if factor == 0: return str(offset)

        representation = variable
        if factor != 1: representation = "%d*%s"%(factor,representation)

        if offset != 0: representation += " + %d"%offset

        return representation

        # return "%s * %s + %s"%(str(factor),
        #                        str(variable),
        #                        str(offset))



