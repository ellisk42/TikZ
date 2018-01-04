from DSL import *
from graphicsSearch import serializeLine

from dispatch import dispatch

import time
import random
import numpy as np

import cPickle as pickle
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.optim as optimization
import torch.cuda as cuda
from torch.nn.utils.rnn import pack_padded_sequence

GPU = cuda.is_available()



def variable(x, volatile=False):
    if isinstance(x,list): x = np.array(x)
    if isinstance(x,(np.ndarray,np.generic)): x = torch.from_numpy(x)
    if GPU: x = x.cuda()
    return Variable(x, volatile=volatile)

LEXICON = ["START","END",
           "circle",
           "rectangle",
           "line","arrow = True","arrow = False","solid = True","solid = False",
           "for",
           "reflect","x","y",
           "}",
           "if",
           "i","j","k","None"] + map(str,range(-5,20))
symbolToIndex = dict(zip(LEXICON,range(len(LEXICON))))

@dispatch(Loop)
def serializeProgram(l):
    return serializeLine(l) + ([] if l.boundary == None else ["if"] + serializeProgram(l.boundary) + ["}"]) + \
        serializeProgram(l.body) + ["}"]
@dispatch(Reflection)
def serializeProgram(l):
    return serializeLine(l) + serializeProgram(l.body) + ["}"]
@dispatch(Primitive)
def serializeProgram(l): return serializeLine(l)
@dispatch(Block)
def serializeProgram(l):
    return [ c for x in l.items for c in serializeProgram(x) ]

def parseOutput(l):
    def get(l):
        n = l[0]
        del l[0]
        return n
    
    def parseLinear(l):
        b = int(get(l))
        x = get(l)
        m = int(get(l))
        if x == 'None': x = None
        return LinearExpression(m,x,b)

    def parseBody(l):
        items = []
        while True:
            if l == []: return Block(items)
            if l[0] == "}":
                get(l)
                return Block(items)
            items.append(parseAtomic(l))
    
    def parseAtomic(l):
        k = get(l)
        if k == 'circle':
            x = parseLinear(l)
            y = parseLinear(l)
            return Primitive(k,x,y)
        if k == 'rectangle':
            x1 = parseLinear(l)
            y1 = parseLinear(l)
            x2 = parseLinear(l)
            y2 = parseLinear(l)
            return Primitive(k,x1,y1,x2,y2)
        if k == 'line':
            x1 = parseLinear(l)
            y1 = parseLinear(l)
            x2 = parseLinear(l)
            y2 = parseLinear(l)
            a = get(l)
            s = get(l)
            return Primitive(k,x1,y1,x2,y2,
                             "arrow = True" == a,
                             "solid = True" == s)
        if k == 'for':
            v = get(l)
            b = parseLinear(l)
            if l[0] == "if":
                get(l)
                boundary = parseBody(l)
            else: boundary = None
            body = parseBody(l)
            return Loop(v = v, bound = b, boundary = boundary, body = body)
        if k == 'reflect':
            a = get(l)
            c = int(get(l))
            body = parseBody(l)
            return Reflection(body = body, axis = a, coordinate = c)
        raise Exception('parsing line '+k)

    return parseBody(l)


class CaptionEncoder(nn.Module):
    def __init__(self):
        super(CaptionEncoder, self).__init__()
        
        (squareFilters,rectangularFilters,numberOfFilters,kernelSizes,poolSizes,poolStrides) = (20,2,[10],
                          [9,9],
                          [8,4],
                          [4,4])
        
        self.squareFilters = nn.Conv2d(1, squareFilters, kernelSizes[0], padding = kernelSizes[0]/2)
        self.verticalFilters = nn.Conv2d(1, rectangularFilters,
                                         (kernelSizes[0]/2 - 1,kernelSizes[0]*2 - 1),
                                         padding = (kernelSizes[0]/4 - 1,kernelSizes[0] - 1))
        self.horizontalFilters = nn.Conv2d(1, rectangularFilters,
                                           (kernelSizes[0]*2 - 1,kernelSizes[0]/2 - 1),
                                           padding = (kernelSizes[0] - 1,kernelSizes[0]/4 - 1))
        self.laterStages = nn.Sequential(nn.ReLU(),
                                         nn.MaxPool2d(poolSizes[0],poolStrides[0],padding = poolSizes[0]/2 - 1),
                                         nn.Conv2d(squareFilters + 2*rectangularFilters,
                                                   numberOfFilters[0],
                                                   kernelSizes[1],
                                                   padding = kernelSizes[1]/2),
                                         nn.ReLU(),
                                         nn.MaxPool2d(poolSizes[1],poolStrides[1],padding = poolSizes[1]/2 - 1))
        

    def forward(self,x):
        c1 = self.squareFilters(x)
        c2 = self.verticalFilters(x)
        c3 = self.horizontalFilters(x)
        c0 = torch.cat((c1,c2,c3),dim = 1)
        output = self.laterStages(c0)
        return output
        #return output.view(output.size(0),-1)

class CaptionDecoder(nn.Module):
    def __init__(self):
        super(CaptionDecoder, self).__init__()
        
        IMAGEFEATURESIZE = 2560
        EMBEDDINGSIZE = 64
        INPUTSIZE = IMAGEFEATURESIZE + EMBEDDINGSIZE
        HIDDEN = 1024
        LAYERS = 2

        # self.embedding : list of N indices (BxW) -> (B,W,EMBEDDINGSIZE)
        self.embedding = nn.Embedding(len(LEXICON),EMBEDDINGSIZE)

        # The embedding is combined with the image features at each time step
        self.rnn = nn.LSTM(INPUTSIZE, HIDDEN, LAYERS, batch_first = True)

        self.tokenPrediction = nn.Linear(HIDDEN,len(LEXICON))

    def forward(self, features, captions, lengths):
        # flatten the convolution output
        features = features.view(features.size(0),-1)
        
        e = self.embedding(captions) # e: BxLx embeddingSize
        #print "e = ",e.size()
        #expandedFeatures: BxTx2560
        expandedFeatures = features.unsqueeze(1).expand(features.size(0),e.size(1),features.size(1))

        #recurrentInputs: bxtxINPUTSIZE
        recurrentInputs = torch.cat((expandedFeatures,e),2)
        #print "recurrentInputs = ",recurrentInputs.size()

        
        packed = pack_padded_sequence(recurrentInputs, lengths, batch_first = True)
        hidden,_ = self.rnn(packed)
        outputs = self.tokenPrediction(hidden[0])
        #print "outputs = ",outputs.size()
        return outputs

    def sample(self, features):
        result = ["START"]

        # (1,1,F)
        features = features.view(-1).unsqueeze(0).unsqueeze(0)
        #features: 1x1x2560
        
        states = None

        while True:
            e = self.embedding(variable([symbolToIndex[result[-1]]]).view((1,-1)))
            recurrentInput = torch.cat((features,e),2)
            output, states = self.rnn(recurrentInput,states)
            distribution = self.tokenPrediction(output).view(-1)
            distribution = F.log_softmax(distribution).data.exp()
            draw = torch.multinomial(distribution,1)[0]
            c = LEXICON[draw]
            if len(result) > 20 or c == "END":
                return result[1:]
            else:
                result.append(c)
            
            

    def buildCaptions(self,tokens):
        '''returns inputs, sizes, targets'''
        
        #tokens = [ [self.symbolToIndex["START"]] + [ self.symbolToIndex[s] for s in serializeProgram(p) ] + [self.symbolToIndex["END"]]
        #          for p in programs ]
        
        # The full token sequences are START, ..., END
        # Training input sequences are START, ...
        # Target output sequences are ..., END
        # the sizes are actually one smaller therefore

        # Make sure that the token sequences are decreasing in size
        previousLength = None
        for t in tokens:
            assert previousLength == None or len(t) <= previousLength
            previousLength = len(t)
            
        
        sizes = map(lambda t: len(t) - 1,tokens)
        maximumSize = max(sizes)
        tokens = [ np.concatenate((p, np.zeros(maximumSize + 1 - len(p),dtype = np.int)))
                   for p in tokens ]
        tokens = np.array(tokens)
        return variable(tokens[:,:-1]),sizes,variable(tokens[:,1:])

class NoTrace(nn.Module):
    def __init__(self):
        super(NoTrace, self).__init__()
        self.encoder = CaptionEncoder()
        self.decoder = CaptionDecoder()

    def sampleMany(self, sequence, duration):
        image = variable(np.array([ sequence.draw() ], dtype = np.float32), volatile = True).unsqueeze(1)

        startTime = time()
        
        imageFeatures = self.encoder(image)
        #imageFeatures: 1x10x16x16

        programs = []
        while time() < startTime + duration:
            nextSequence = self.decoder.sample(imageFeatures)
            try:
                p = parseOutput(nextSequence)
                print "Sampled",p
                programs.append((p,p.convertToSequence()))
            except: continue
        return programs
    

    def loss(self,examples):
        # IMPORTANT: Sort the examples by their size. recurrent network stuff needs this
        examples.sort(key = lambda e: len(e.tokens), reverse = True)
        
        x = variable(np.array([ e.sequence.draw() for e in examples], dtype = np.float32))

        x = x.unsqueeze(1) # insert the channel

        imageFeatures = self.encoder(x)
        
        inputs, sizes, T = self.decoder.buildCaptions([ e.tokens for e in examples ])
        
        outputDistributions = self.decoder(imageFeatures, inputs, sizes)
        
        T = pack_padded_sequence(T, sizes, batch_first = True)[0]
        
        return F.cross_entropy(outputDistributions, T)

    def load(self,path):
        if os.path.isfile(path):
            if not GPU: stuff = torch.load(path,map_location = lambda s,l: s)
            else: stuff = torch.load(path)
            self.load_state_dict(stuff)
            print "Loaded checkpoint",path
        else:
            print "Could not find checkpoint",path

    def dump(self,path):
        torch.save(self.state_dict(),path)
        print "Dumped checkpoint",path

    
class TrainingExample():
    def __init__(self,p):
        try:
            self.tokens = np.array([symbolToIndex["START"]] + [ symbolToIndex[s] for s in serializeProgram(p) ] + [symbolToIndex["END"]])
        except KeyError:
            print "Key error in tokenization",serializeProgram(p)
            assert False
        
        self.sequence = p.convertToSequence()
        #self.program = p

        if str(parseOutput(serializeProgram(p))) != str(p):
            print "Serialization failure for program",p
            print serializeProgram(p)
            print parseOutput(serializeProgram(p))
            assert False

def loadTrainingData(n):
    print "About to load the examples"
    alternatives = ['/scratch/ellisk/randomlyGeneratedPrograms.p',
                    'randomlyGeneratedPrograms.p']
    for alternative in alternatives:
        if os.path.exists(alternative):
            trainingDataPath = alternative
            print "Loading training data from",trainingDataPath
            break
        
    with open(trainingDataPath,'rb') as handle:
        X = pickle.load(handle)
    print "Keeping %d/%d examples"%(n,len(X))
    pruned = []
    
    for x in X:
        x = pickle.loads(x)
        if x.items != []:
            pruned.append(TrainingExample(x))
        if len(pruned) >= n:
            break
    print "Pruned down to %d examples"%(len(pruned))
    return pruned
        
if __name__ == "__main__":
    import sys
    
    model = NoTrace()
    if GPU:
        print "Using the GPU"
        model = model.float().cuda()
    else:
        print "Using the CPU"
        model = model.float()

    model.load("checkpoints/noTrace.torch")

    if 'test' in sys.argv:
        from groundTruthParses import *
        import os
        target = getGroundTruthParse('drawings/expert-%s.png'%(sys.argv[2]))
        results = model.sampleMany(target, 60*60)
        results.sort(key = lambda (_,s): s - target)
        if len(results) > 0:
            (p,s) = results[0]
            print "Best program:"
            print p.pretty()
            #showImage(np.concatenate((1 - target.draw(),s.draw()),axis = 1))
            saveMatrixAsImage(255*np.concatenate((1 - target.draw(),s.draw()),axis = 0),
                              "noTraceOutputs/%s.png"%(sys.argv[2]))

        os.exit(0)
        
    #print "# Learnable parameters:",sum([ parameter.view(-1).shape[0] for parameter in model.parameters() ])
        

    N = 1*(10**7)
    B = 64
    X = loadTrainingData(N)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    E = 0
    while True:
        E += 1
        print "epic",E
        # scrambled the data
        X = list(np.random.permutation(X))
        start = 0
        batchesPerLoop = N/B
        batchIndex = 0
        while start < N:
            batch = X[start:start + B]
            
            model.zero_grad()
            L = model.loss(batch)
            if batchIndex%50 == 0:
                print "Batch [%d/%d], LOSS = %s"%(batchIndex,batchesPerLoop,L.data[0])
                model.dump("checkpoints/noTrace.torch")
                    
            L.backward()
            optimizer.step()

            start += B
            batchIndex += 1
