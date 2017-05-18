import re
import numpy as np
import cv2
from utilities import loadImage,showImage
from subprocess import Popen, PIPE
import os
import tempfile

from language import *

def synthesizeProgram(parse,usePrior = True):
    parts = []
    hasCircles = False
    hasRectangles = False
    hasLines = False
    noDiagonals = True
    arrows = [] #}settings of the arrow parameters that are observed in the data
    solid = [] # settings of the solid parameters are so served in the data
    
    if parse.lines == []:
        return None

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
            if p.isDiagonal(): noDiagonals = False
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

%s    
#define MAXIMUMLOOPITERATIONS 4
#define MAXIMUMXCOORDINATE %d
#define MAXIMUMYCOORDINATE %d
#define HASCIRCLES %d
#define HASRECTANGLES %d
#define HASLINES %d
#define HASSOLID %d
#define HASDASHED %d
#define HASARROW %d
#define HASNOARROW %d
#define NODIAGONALS %d

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
     ('#define USEPRIOR' if usePrior else ''),
     x1,y1,
     hasCircles,hasRectangles,hasLines,
     (True in solid),(False in solid),
     (True in arrows),(False in arrows),
     int(noDiagonals),
     " || ".join(parts))

    fd = tempfile.NamedTemporaryFile(mode = 'w',suffix = '.sk',delete = False,dir = '.')
    fd.write(source)
    fd.close()

    # Temporary file for collecting the sketch output
    od = tempfile.NamedTemporaryFile(mode = 'w',delete = False,dir = '/tmp')
    od.write('') # just create the file that were going to overwrite
    od.close()
    outputFile = od.name

    os.system('sketch --fe-timeout 180 -V 10 %s 2> %s > %s'%(fd.name, outputFile, outputFile))

    output = open(outputFile,'r').read()
    os.remove(fd.name)
    os.remove(outputFile)

    # Recover the program length from the sketch output
    if usePrior:
        programSize = [ l for l in output.split('\n') if "*********INSIDE minimizeHoleValue" in l ] #if () {}
        if programSize == []:
            print "Synthesis failure!"
            print output
            return None
        programSize = programSize[-1]
        m = re.match('.*=([0-9]+),',programSize)
        cost = int(m.group(1))
    else:
        cost = -1
    
    # find the body of the synthesized code
    body = None
    for l in output.split('\n'):
        if body == None and 'void render ' in l:
            body = [l]
        elif body != None:
            body.append(l)
            if 'minimize' in l:
                break
    if body != None:
        body = "\n".join(body)
    else:
        print "WARNING: Could not parse body."
        # print body
    # parseSketchOutput(body)
    # assert False
    return cost,body
