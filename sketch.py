from utilities import log2
from math import ceil,log
import re
import numpy as np
from utilities import loadImage,showImage
from subprocess import Popen, PIPE
import os
import tempfile

from language import *

def possibleCoefficients(parse):
    x = []
    y = []
    for p in parse.lines[:-1]:
        for q in parse.lines[1:]:
            if p == q: continue
            if isinstance(p,Circle) and isinstance(q,Circle):
                x.append(p.center.x - q.center.x)
                y.append(p.center.y - q.center.y)
            if isinstance(p,Rectangle) and isinstance(q,Rectangle):
                x.append(p.p1.x - q.p1.x)
                x.append(p.p2.x - q.p2.x)
                y.append(p.p1.y - q.p1.y)
                y.append(p.p2.y - q.p2.y)
            if isinstance(p,Line) and isinstance(q,Line):
                if p.solid != q.solid or p.arrow != q.arrow: continue
                x.append(p.points[0].x - q.points[0].x)
                x.append(p.points[1].x - q.points[1].x)
                y.append(p.points[0].y - q.points[0].y)
                y.append(p.points[1].y - q.points[1].y)
    return set(x) - set([0]),set(y) - set([0])

def synthesizeProgram(parse,usePrior = True,entireParse = None,
                      xCoefficients = [],
                      yCoefficients = [],
                      usedReflections = [],
                      usedLoops = [],
                      CPUs = 1,
                      maximumDepth = 3,
                      canReflect = True,
                      canLoop = True):
    ''' usedLoops: [{depth, coefficient, variable, intercept}]'''
    
    parts = []
    hasCircles = False
    hasRectangles = False
    hasLines = False
    noDiagonals = True
    arrows = [] #}settings of the arrow parameters that are observed in the data
    solid = [] # settings of the solid parameters are so served in the data
    
    if parse.lines == []:
        return None

    if entireParse == None: entireParse = parse

    # translate drawing the lower left-hand corner
    x0 = min([x for l in entireParse.lines for x in l.usedXCoordinates()  ])
    y0 = min([y for l in entireParse.lines for y in l.usedYCoordinates()  ])
    x1 = max([x for l in entireParse.lines for x in l.usedXCoordinates()  ]) - x0
    y1 = max([y for l in entireParse.lines for y in l.usedYCoordinates()  ]) - y0
    biggestNumber = -1 #max([x1,y1,4])

    # all of the allowed X and Y coordinates
    yValidation = []
    xValidation = []
    
    for p in parse.lines:
        if isinstance(p,Circle):
            parts.append("_c(%d,%d)"%(p.center.x - x0,
                                      p.center.y - y0))
            xValidation.append(p.center.x - x0)
            yValidation.append(p.center.y - y0)
            hasCircles = True
        elif isinstance(p,Rectangle):
            hasRectangles = True
            parts.append("_r(%d,%d,%d,%d)"%(p.p1.x - x0,
                                            p.p1.y - y0,
                                            p.p2.x - x0,
                                            p.p2.y - y0))
            xValidation.append(p.p1.x - x0)
            yValidation.append(p.p1.y - y0)
            xValidation.append(p.p2.x - x0)
            yValidation.append(p.p2.y - y0)
        elif isinstance(p,Line):
            hasLines = True
            if p.isDiagonal(): noDiagonals = False
            arrows.append(p.arrow)
            solid.append(p.solid)
            parts.append("_l(%d,%d,%d,%d,%d,%d)"%(p.points[0].x - x0,
                                                  p.points[0].y - y0,
                                                  p.points[1].x - x0,
                                                  p.points[1].y - y0,
                                                  0 if p.solid else 1,
                                                  1 if p.arrow else 0))
            xValidation.append(p.points[0].x - x0)
            yValidation.append(p.points[0].y - y0)
            xValidation.append(p.points[1].x - x0)
            yValidation.append(p.points[1].y - y0)

    coefficientGenerator1 = ''
    for c in xCoefficients: coefficientGenerator1 = '| ' + str(c)
    coefficientGenerator2 = ''
    for c in yCoefficients: coefficientGenerator2 = '| ' + str(c)

    coefficientValidator1,coefficientValidator2 = possibleCoefficients(parse)
    coefficientValidator1 = " || ".join([ "c == %d"%c for c in coefficientValidator1 ])
    if len(coefficientValidator1) == 0: coefficientValidator1 = 'numberOfCoefficients1 == 0'
    coefficientValidator2 = " || ".join([ "c == %d"%c for c in coefficientValidator2 ])
    if len(coefficientValidator2) == 0: coefficientValidator2 = 'numberOfCoefficients2 == 0'

    xValidation = " || ".join([ "x == %d"%x for x in set(xValidation) ])
    yValidation = " || ".join([ "x == %d"%x for x in set(yValidation) ])

    haveThisReflectionAlready = " || ".join([ "(xr == %d && yr == %d)"%(xr,yr)
                                              for xr,yr in usedReflections ] + ['0'])

    alreadyProvidedBounds = ''
    for bound in usedLoops:
        returnValue = str(bound['intercept'])
        if bound['coefficient'] != None and bound['coefficient'] != 0:
            returnValue += ' + %s*environment[%d]'%(bound['coefficient'],bound['variable'])
        alreadyProvidedBounds += ' if (n == %d && ??) { already_have_this_loop = 1; return %s; }'%(bound['depth'],
                                                                                             returnValue)
    # print "alreadyProvidedBounds:"
    # print alreadyProvidedBounds

    smallestPossibleLoss = 3*len(parse.lines) + 1
    upperBoundOnLoss = ' --bnd-mbits %d'%(min(5,int(ceil(log2(smallestPossibleLoss + 1)))))
    
    source = '''
pragma options "--bnd-unroll-amnt 4 --bnd-arr1d-size 2 --bnd-arr-size 2 --bnd-int-range %d %s";

%s    
#define MAXIMUMDEPTH %d
#define CANLOOP %d
#define CANREFLECT %d
#define ALREADYPROVIDEDBOUNDS %s
#define HAVETHISREFLECTIONALREADY %s
#define XCOEFFICIENTS %s
#define YCOEFFICIENTS %s
#define PROVIDEDXCOEFFICIENTS %d
#define PROVIDEDYCOEFFICIENTS %d
#define XVALIDATION ( %s )
#define YVALIDATION ( %s )
#define COEFFICIENTVALIDATOR1 ( %s )
#define COEFFICIENTVALIDATOR2 ( %s )
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

#define COSTUPPERBOUND %d

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
'''%(biggestNumber, upperBoundOnLoss,
     ('#define USEPRIOR' if usePrior else ''),
     maximumDepth,
     int(canLoop),int(canReflect),
     alreadyProvidedBounds,
     haveThisReflectionAlready,
     coefficientGenerator1, coefficientGenerator2,
     len(xCoefficients), len(yCoefficients),
     xValidation,yValidation,
     coefficientValidator1,coefficientValidator2,
     x1,y1,
     hasCircles,hasRectangles,hasLines,
     (True in solid),(False in solid),
     (True in arrows),(False in arrows),
     int(noDiagonals),
     len(parse.lines),
     " || ".join(parts))

    print source

    fd = tempfile.NamedTemporaryFile(mode = 'w',suffix = '.sk',delete = False,dir = '.')
    fd.write(source)
    fd.close()

    # Temporary file for collecting the sketch output
    od = tempfile.NamedTemporaryFile(mode = 'w',delete = False,dir = '/tmp')
    od.write('') # just create the file that were going to overwrite
    od.close()
    outputFile = od.name

    degreeOfParallelism = ''
    if CPUs != 1:
        degreeOfParallelism = '--slv-parallel --slv-p-cpus %d'%CPUs

    os.system('sketch %s --fe-timeout 180 -V 10 %s 2> %s > %s'%(degreeOfParallelism,
                                                                fd.name, outputFile, outputFile))

    output = open(outputFile,'r').read()
    os.remove(fd.name)
    os.remove(outputFile)

    # Recover the program length from the sketch output
    if usePrior:
        programSize = [ l for l in output.split('\n') if "*********INSIDE minimizeHoleValue" in l ] #if () {}
        if programSize == []:
            # print "Synthesis failure!"
            # print output + "\n" + source
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
        fd = tempfile.NamedTemporaryFile(mode = 'w',suffix = '.failure',delete = False,dir = './parsingErrors')
        fd.write(output)
        fd.close()
        print "WARNING: Could not parse body. Leaving the unparsable body in %s"%(fd.name)
        
        # print body
    # parseSketchOutput(body)
    # assert False
    return cost,body
