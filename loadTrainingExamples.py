from utilities import *
from language import *

import os
import io
import pickle
import tarfile

def loadTar(f = 'syntheticTrainingData.tar'):
    if os.path.isfile('/om/user/ellisk/%s'%f):
        handle = '/om/user/ellisk/%s'%f
    else:
        handle = f
    print "Loading data from",handle
    handle = tarfile.open(handle)
    
    # just load everything into RAM - faster that way. screw you tar
    members = {}
    for member in handle:
        if member.name == '.': continue
        stuff = handle.extractfile(member)
        members[member.name] = stuff.read()
        stuff.close()
    handle.close()

    print "Loaded tar file into RAM: %d entries."%len(members)
    return members


def loadExamples(numberOfExamples, f = 'syntheticTrainingData.tar'):
    members = loadTar(f)
    programNames = [ "./randomScene-%d.p"%(j)
                     for j in range(numberOfExamples) ]
    programs = [ pickle.load(io.BytesIO(members[n])) for n in programNames ]

    print "Loaded pickles."

    noisyTarget = [ "./randomScene-%d-noisy.png"%(j) for j in range(numberOfExamples) ]
    for t in noisyTarget:
        cacheImage(t,members[t])

    print "Loaded images."

    return noisyTarget, programs


# def makeTraceExamples(numberOfExamples, noisy = True):
#     noisyTargets, programs = loadExamples(numberOfExamples)

#     startingExamples = []
#     endingExamples = []
#     targetExamples = []

#     # get one example from each line of each program
#     for j,program in enumerate(programs):
#         if j%10000 == 1:
#             print "Processed %d/%d programs"%(j - 1,len(programs))
#         noisyTarget = "./randomScene-%d-noisy.png"%(j) if noisyTrainingData else trace[-1]
#         # cache the images
#         for imageFilename in [noisyTarget] + trace:
#             cacheImage(imageFilename, members[imageFilename])
#         if not dummyImages:
#             trace = loadImages(trace)
#             noisyTarget = loadImage(noisyTarget)
        
#         targetImage = trace[-1]
#         currentImage = "blankImage" if dummyImages else np.zeros(targetImage.shape)
#         for k,l in enumerate(program.lines):
#             startingExamples.append(currentImage)
#             endingExamples.append(noisyTarget)
#             targetLine.append(l)
#             currentImage = trace[k]
#             for j,t in enumerate(PrimitiveDecoder.extractTargets(l)):
#                 if not j in target: target[j] = []
#                 target[j].append(t)
#         # end of program
#         startingExamples.append(targetImage)
#         endingExamples.append(noisyTarget)
#         targetLine.append(None)
#         for j in target:
#             target[j] += [STOP] # should be zero and therefore valid for everyone
            
#     targetVectors = [np.array(target[j]) for j in sorted(target.keys()) ]

#     print "loaded images in",(time() - startTime),"s"
#     print "target dimensionality:",len(targetVectors)

#     return np.array(startingExamples), np.array(endingExamples), targetVectors, np.array(targetLine)
