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
    print source

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
    return cost,body

def parseSketchOutput(output):
    for l in output.split('\n'):
        pattern = '\(\(\(shapeIdentity == 0\) && \(cx == (.+)\)\) && \(cy == (.+)\)\)'
        m = re.findall(pattern,l)
        assert len(m) < 2
        if m != []:
            x = parseExpression(m[0][0])
            y = parseExpression(m[0][1])
            print "Circle(%s,%s)"%(x,y)
            continue

        pattern = '\(\(\(\(\(shapeIdentity == 2\) && \((.+) == rx1\)\) && \((.+) == ry1\)\) && \((.+) == rx2\)\) && \((.+) == ry2\)\)'
        m = re.search(pattern,l)
        if m:
            print "Rectangle(%s,%s,%s,%s)"%(parseExpression(m.group(1)),
                                            parseExpression(m.group(2)),
                                            parseExpression(m.group(3)),
                                            parseExpression(m.group(4)))
            continue

        pattern = 'for\(int .* = 0; .* < (.*); .* = .* \+ 1\)'
        m = re.search(pattern,l)
        if m:
            print "for (%s)"%(parseExpression(m.group(1)))
            continue
        
def parseExpression(e):
    try: return int(e)
    except:
        factor = re.search('([\-0-9]+) * ',e)
        if factor != None: factor = int(factor.group(1))
        offset = re.search(' \+ ([\-0-9]+)',e)
        if offset != None: offset = int(offset.group(1))
        variable = re.search('\[(\d)\]',e)
        if variable != None: variable = ['i','j'][int(variable.group(1))]

        return "%s * %s + %s"%(str(factor),
                               str(variable),
                               str(offset))



# parseSketchOutput('''
# void render (int shapeIdentity, int cx, int cy, int lx1, int ly1, int lx2, int ly2, bit dashed, bit arrow, int rx1, int ry1, int rx2, int ry2, ref bit _out)  implements renderSpecification/*tmpWsAXFS.sk:179*/
# {
#   _out = 0;
#   assume (((shapeIdentity == 0) || (shapeIdentity == 1)) || (shapeIdentity == 2)): "Assume at tmpWsAXFS.sk:180"; //Assume at tmpWsAXFS.sk:180
#   assume (shapeIdentity != 1): "Assume at tmpWsAXFS.sk:183"; //Assume at tmpWsAXFS.sk:183
#   int[0] environment = {};
#   int loop_body_cost = 0;
#   bit _pac_sc_s6_s8 = 0;
#   for(int j = 0; j < 4; j = j + 1)/*Canonical*/
#   {
#     bit _pac_sc_s17 = _pac_sc_s6_s8;
#     if(!(_pac_sc_s6_s8))/*tmpWsAXFS.sk:79*/
#     {
#       int[1] _pac_sc_s17_s19 = {0};
#       push(0, environment, j, _pac_sc_s17_s19);
#       bit _pac_sc_s17_s21 = 0 || (((((shapeIdentity == 2) && (((-3 * (_pac_sc_s17_s19[0])) + 9) == rx1)) && (((2 * (_pac_sc_s17_s19[0])) + -2) == ry1)) && (((-3 * (_pac_sc_s17_s19[0])) + 11) == rx2)) && (6 == ry2));
#       bit _pac_sc_s6 = _pac_sc_s17_s21;
#       int newCost_0 = 1;
#       if(!(_pac_sc_s17_s21))/*tmpWsAXFS.sk:59*/
#       {
#         int loop_body_cost_0 = 0;
#         bit _pac_sc_s6_s8_0 = 0;
#         for(int j_0 = 0; j_0 < (_pac_sc_s17_s19[0]); j_0 = j_0 + 1)/*Canonical*/
#         {
#           bit _pac_sc_s17_0 = _pac_sc_s6_s8_0;
#           if(!(_pac_sc_s6_s8_0))/*tmpWsAXFS.sk:79*/
#           {
#             int[2] _pac_sc_s17_s19_0 = {0,0};
#             push(1, _pac_sc_s17_s19, j_0, _pac_sc_s17_s19_0);
#             loop_body_cost_0 = 1;
#             _pac_sc_s17_0 = 0 || (((shapeIdentity == 0) && (cx == ((3 * (_pac_sc_s17_s19_0[0])) + -2))) && (cy == ((-2 * (_pac_sc_s17_s19_0[1])) + 5)));
#           }
#           _pac_sc_s6_s8_0 = _pac_sc_s17_0;
#         }
#         newCost_0 = loop_body_cost_0 + 1;
#         _pac_sc_s6 = _pac_sc_s6_s8_0;
#       }
#       loop_body_cost = 1 + newCost_0;
#       _pac_sc_s17 = _pac_sc_s6;
#     }
#     _pac_sc_s6_s8 = _pac_sc_s17;
#   }
#   _out = _pac_sc_s6_s8;
#   minimize(loop_body_cost + 1)''')
