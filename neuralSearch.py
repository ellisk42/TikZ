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
    def __init__(self, lexicon, H = 32, layers = 1, seedDimensionality = 32):
        super(self.__class__,self).__init__()

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

    def sample(self, seed, maximumLength):
        h = self.h0(seed).view(self.layers, 1, self.H)

        accumulator = ["START"]
        for _ in range(maximumLength):
            i = self.targetsOfSymbols([accumulator[-1]])[:,0]
            output, h = self(i,h)
            distribution = output.data.view(-1).exp()

            c = torch.multinomial(distribution,1)[0]
            if self.lexicon[c] == "END": break
            
            accumulator.append(self.lexicon[c])
            
        return accumulator[1:]            
        
            

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
        self.lineDecoder = LineDecoder(lexicon, H = H)
        self.circleEncoder = DataEncoder(2, H)

    def encodeProblem(self, s):
        return sum(\
                self.circleEncoder(Variable(torch.from_numpy(np.array([c.center.x, c.center.y])).float())) \
                for c in s)

    def loss(self, s, target, criteria):
        e = self.encodeProblem(s).view(1,-1)

        return self.lineDecoder.loss(target, e, criteria)

    def sample(self, s):
        return self.lineDecoder.sample(self.encodeProblem(s).view(1,-1), 10)
        
        

if __name__ == "__main__":
    lexicon = ["START","END","CIRCLE"] + map(str,range(1,16))
    p = SearchPolicy(lexicon)
    o = optimization.Adam(p.parameters(), lr = 0.001)
    criteria = nn.CrossEntropyLoss()
    
    for step in range(10000):
        x = random.choice(range(1,4))
        y = random.choice(range(1,4))
        scene = [Circle.absolute(x,y)]
        target = ["CIRCLE",str(x),str(y),"END"]

        o.zero_grad()
        loss = p.loss(scene, target, criteria)
        loss.backward()
        o.step()

        if step%100 == 0:
            print step,'\t',loss.data[0]
            print scene[0]
            print p.sample(scene)
#            print p.beam(p.encodeScene(scene))[0][1],scene[0]

    
        
        
