from language import *

import os
import io
import pickle
import tarfile
from utilities import *
import loadTrainingExamples

import sys
arguments = sys.argv
if len(arguments) > 1:
    outputDirectory = arguments[1]
else:
    outputDirectory = '../CycleGAN-tensorflow/datasets/sketchy'

examples = loadTrainingExamples.loadTar()
j = 0
for k in examples:
    print(k)
    if k.endswith('.p'):
        saveMatrixAsImage(255*(1 - pickle.load(io.BytesIO(examples[k])).draw()), '%s/trainA/%d.png'%(outputDirectory,j))
        saveMatrixAsImage(255*(1 - pickle.load(io.BytesIO(examples[k])).draw()), '%s/testA/%d.png'%(outputDirectory,j))
        j += 1
