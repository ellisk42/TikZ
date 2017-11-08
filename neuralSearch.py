from DSL import *

from dispatch import dispatch

import random
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.optim as optimization

class LineDecoder(nn.Module):
    def __init__(self, lexicon, H = 32, layers = 1, seedDimensionality = None):
        super(self.__class__,self).__init__()

        seedDimensionality = seedDimensionality or H

        assert "START" in lexicon
        assert "END" in lexicon
        
        self.lexicon = lexicon
        self.model = nn.GRU(H,H, layers)

        self.encoder = nn.Embedding(len(lexicon), H)
        self.decoder = nn.Linear(H, len(lexicon))

        self.layers = layers
        self.h0 = nn.Linear(seedDimensionality, H*layers)
                                

        self.H = H

    def forward(self, input, hidden):
        """input: something of size """
        B = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.model(encoded.view(1, B, -1), hidden)
        output = self.decoder(output.view(B, -1))
        return output, hidden

    def targetsOfSymbols(self, symbols):
        B = 1
        t = torch.LongTensor(B,len(symbols))
        for j,s in enumerate(symbols):
            t[0,j] = self.lexicon.index(s)
        return Variable(t)
        

    def loss(self, target, seed, criteria):
        numberOfTargets = len(target)
        i = self.targetsOfSymbols(["START"] + target[:-1])
        target = self.targetsOfSymbols(target)

        B = 1 # batch size
        h = self.h0(seed).view(self.layers, 1, self.H)
        L = 0
        for j in range(numberOfTargets):
            output, h = self(i[:,j],h)
            L += criteria(output.view(B, -1), target[:,j])
        return L

    def sample(self, seed, maximumLength, T = 1):
        h = self.h0(seed).view(self.layers, 1, self.H)

        accumulator = ["START"]
        for _ in range(maximumLength):
            i = self.targetsOfSymbols([accumulator[-1]])[:,0]
            output, h = self(i,h)
            distribution = (output.data.view(-1)/T).exp()

            c = torch.multinomial(distribution,1)[0]
            if self.lexicon[c] == "END": break
            
            accumulator.append(self.lexicon[c])
            
        return accumulator[1:]

class LineEncoder(nn.Module):
    def __init__(self, lexicon, H = 32, layers = 1, seedDimensionality = 32):
        super(self.__class__,self).__init__()

        self.lexicon = lexicon
        self.model = nn.GRU(H,H, layers)

        self.encoder = nn.Embedding(len(lexicon), H)

        self.layers = layers
        self.h0 = nn.Linear(seedDimensionality, H*layers)
                                
        self.H = H

    def forward(self, inputs, seed):
        B = inputs.size(0)
        assert B == seed.size(0)
        
        encoded = self.encoder(inputs).permute(1,0,2)
        hidden = self.h0(seed).view(self.layers, B, self.H)
        
        output, hidden = self.model(encoded, hidden)
        
        return hidden.view(B,-1)

    def tensorOfSymbols(self, symbols):
        B = 1
        t = torch.LongTensor(B,len(symbols))
        for j,s in enumerate(symbols):
            t[0,j] = self.lexicon.index(s)
        return Variable(t)

    def encoding(self, symbols, seed):
        return self(self.tensorOfSymbols(symbols), seed)
            

class DataEncoder(nn.Module):
    """
    A network that takes something like circle(9,2) and produces an embedding
    """
    def __init__(self, numberOfInputs, embeddingSize):
        super(DataEncoder,self).__init__()

        self.linear1 = nn.Linear(numberOfInputs, embeddingSize)

    def forward(self, x):
        return self.linear1(x).clamp(min = 0)

class SearchPolicy(nn.Module):
    def __init__(self, lexicon):
        super(self.__class__,self).__init__()

        H = 32
        self.lineDecoder = LineDecoder(lexicon,
                                       # The line decoder needs to take both environment and problem
                                       H = 2*H)
        self.lineEncoder = LineEncoder([l for l in lexicon if not (l in ['START','END']) ],
                                       H = H)
        self.circleEncoder = DataEncoder(2, H)

        self.environmentScore = nn.Linear(H,1)
        
        

    def encodeProblem(self, s):
        return sum(\
                self.circleEncoder(Variable(torch.from_numpy(np.array([c.center.x, c.center.y])).float())) \
                for c in s)

    def encodeEnvironment(self, environment, seed):
        return self.lineEncoder.encoding(environment, seed)

    def environmentLogLikelihoods(self, environments, problem):
        environmentEncodings = [ self.encodeEnvironment(e, problem)
                                 for e in environments ] 
        environmentScores = [ self.environmentScore(e)
                              for e in environmentEncodings ]
        return torch.nn.functional.log_softmax(torch.cat(environmentScores).view(-1))

    def loss(self, s,
             target, # the target line of code: should be a list of tokens
             environments, # the target environment (first) and the alternatives (following)
             criteria):
        problem = self.encodeProblem(s).view(1,-1)

        environmentLoss =  - self.environmentLogLikelihoods(environments, problem)[0]

        e = self.encodeEnvironment(environments[0], problem)
        seed = torch.cat([problem, e],dim = 1)
        
        return self.lineDecoder.loss(target, seed, criteria) + environmentLoss

    def sampleLine(self, s, e, T = 1):
        problem = self.encodeProblem(s).view(1,-1)
        e = self.encodeEnvironment(e, problem)
        seed = torch.cat([problem, e], dim = 1)
        return self.lineDecoder.sample(seed, 10, T = T)
    def sampleEnvironment(self, s, environments, T = 1):
        problem = self.encodeProblem(s).view(1,-1)
        environmentScores = self.environmentLogLikelihoods(environments, problem)
        distribution = (environmentScores/T).exp()
        i = torch.multinomial(distribution.data, 1)[0]
        return environments[i]
    def sample(self, s, environments, T = 1):
        e = self.sampleEnvironment(s, environments, T = T)
        l = self.sampleLine(s, e, T = T)
        return e,l
        
        

if __name__ == "__main__":
    lexicon = ["START","END","CIRCLE","for","x","y","reflect"] + map(str,range(1,16))
    p = SearchPolicy(lexicon)
    o = optimization.Adam(p.parameters(), lr = 0.001)
    criteria = nn.CrossEntropyLoss()
    
    for step in range(10000):
        x = random.choice(range(1,4))
        y = random.choice(range(1,4))
        scene = [Circle.absolute(x,y)]
        target = ["CIRCLE",str(x),str(y),"END"]

        desiredEnvironment = ["reflect","x","1"]
        alternativeEnvironment = ["for","x","2"]
        environments = [desiredEnvironment,alternativeEnvironment]

        l = p.lineEncoder.tensorOfSymbols(target[:-1])
        h = p.encodeProblem(scene).view(1,-1)

        o.zero_grad()
        loss = p.loss(scene, target, environments, criteria)
        loss.backward()
        o.step()

        if step%100 == 0:
            print step,'\t',loss.data[0]
            print scene[0]
            print p.sample(scene, environments)
            #print p.sample(scene, T = 1)
#            print p.beam(p.encodeScene(scene))[0][1],scene[0]

    
        
        
