import random
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.optim as optimization
import torch.cuda as cuda

GPU = cuda.is_available()

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
        if GPU: t = t.cuda()
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
            distribution = output.data.view(-1)/T
            distribution = F.log_softmax(distribution).data
            distribution = distribution.exp()

            c = torch.multinomial(distribution,1)[0]
            if self.lexicon[c] == "END": break
            
            accumulator.append(self.lexicon[c])
            
        return accumulator[1:]

    def beam(self, seed, maximumLength, beamSize):
        h = self.h0(seed).view(self.layers, 1, self.H)

        B = [(0.0,["START"],h)]
        
        for _ in range(maximumLength):
            expanded = False
            new = []
            for ll,sequence,h in B:
                if sequence[-1] == "END":
                    new.append((ll,sequence,None))
                    continue
                expanded = True
                i = self.targetsOfSymbols([sequence[-1]])[:,0]
                output, h = self(i,h)
                distribution = F.log_softmax(output.data.view(-1)).data
                best = sorted(zip(distribution,self.lexicon),reverse = True)[:beamSize]
                for _ll,c in best:
                    new.append((ll + _ll,
                                sequence + [c],
                                h))
            if not expanded: break
            B = sorted(new,reverse = True)[:beamSize]

        B = [ (ll,sequence[1:-1]) for ll,sequence,h in sorted(B,reverse = True)[:beamSize]
              if sequence[-1] == "END"]
        return B
        

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
        if GPU: t = t.cuda()
        return Variable(t)

    def encoding(self, symbols, seed):
        return self(self.tensorOfSymbols(symbols), seed)
            

class SearchPolicy(nn.Module):
    def __init__(self, lexicon):
        super(SearchPolicy, self).__init__()

        H = 64
        layers = 1
        self.lineDecoder = LineDecoder(lexicon,
                                       layers = 1,
                                       # The line decoder needs to take both environment and problem
                                       H = 2*H,
                                       seedDimensionality = layers*H*2)
        self.lineEncoder = LineEncoder(lexicon,
                                       layers = 1,
                                       H = H,
                                       seedDimensionality = H)

        self.environmentScore = nn.Linear(H,1)

        self.H = H
        
        

    def encodeEnvironment(self, environment, seed):
        return self.lineEncoder.encoding(["START"] + environment, seed)

    def environmentLogLikelihoods(self, environments, problem):
        environmentEncodings = [ self.encodeEnvironment(e, problem)
                                 for e in environments ] 
        environmentScores = [ self.environmentScore(e)
                              for e in environmentEncodings ]
        return F.log_softmax(torch.cat(environmentScores).view(-1))

    def loss(self, example, criteria):
        problem = self.encodeProblem(example.problem).view(1,-1)

        environmentLoss =  - self.environmentLogLikelihoods(example.environments, problem)[0]

        e = self.encodeEnvironment(example.environments[0], problem)
        seed = torch.cat([problem, e],dim = 1)
        
        return self.lineDecoder.loss(example.target + ["END"], seed, criteria) + environmentLoss

    def makeOracleExamples(self, program, problem):
        examples = []
        for intermediateProgram, environment, target in self.Oracle(program):
            intermediateProblem = self.residual(problem, self.evaluate(intermediateProgram))
            environments = self.candidateEnvironments(intermediateProgram)
            environments.remove(environment)
            environments = [environment] + environments
            ex = PolicyTrainingExample(problem, target, environments)
            examples.append(ex)
            
        return examples

    def mayBeAppliedChange(self, initialProgram, environment, line):
        try: return self.applyChange(initialProgram, environment, line)
        except: return None

    def sampleLine(self, s, e, T = 1):
        problem = self.encodeProblem(s).view(1,-1)
        e = self.encodeEnvironment(e, problem)
        seed = torch.cat([problem, e], dim = 1)
        return self.lineDecoder.sample(seed, 10, T = T)
    def beamLine(self, problem, environment, beamSize):
        problem = self.encodeProblem(problem).view(1,-1)
        e = self.encodeEnvironment(environment, problem)
        seed = torch.cat([problem, e], dim = 1)
        return self.lineDecoder.beam(seed, 20, beamSize)
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
    def beam(self, problem, initialProgram, size):
        environments = self.candidateEnvironments(initialProgram)
        environmentScores = self.environmentLogLikelihoods(environments,
                                                           self.encodeProblem(problem).view(1,-1)).data
        candidates = [ (environmentScore + lineScore,
                        self.mayBeAppliedChange(initialProgram, environment, line))
            for environment, environmentScore in zip(environments, environmentScores)
            for lineScore, line in self.beamLine(problem, environment, size)  ]
        candidates = [ (score, candidate) for (score, candidate) in candidates if candidate != None ]
        candidates = list(sorted(candidates, reverse = True)[:size])
        return candidates

    def beamSearchGraph(self, problem, initialProgram, size, steps):
        frontier = [initialProgram]

        for step in range(steps):
            newFrontier = []
            for f in frontier:
                for _,candidate in self.beam(self.residual(problem, self.evaluate(f)),
                                             f, size):
                    print "STEP = %s; PARENT = %s; CHILD = %s;"%(step,f,candidate)
                    newFrontier.append(candidate)
            #newFrontier = removeDuplicateStrings(newFrontier)
            newFrontier = [(self.value(problem,f),f) for f in newFrontier ]
            newFrontier.sort(reverse = True)
            print "New frontier:"
            for v,f in newFrontier: print "V = ",v,"\t",f
            if self.solvesTask(problem, f):
                print "SOLVED TASK!"
                return 
            print "(end of new frontier)"
            print 
            # import pdb
            # pdb.set_trace()
            
            frontier = [ f for v,f in newFrontier[:size] ]


            print "Step %d of graph search:"%step
            for f in frontier: print f
            print "(end of step)"
            print 
        

    def sampleOneStep(self, problem, initialProgram):
        problem = self.residual(problem, self.evaluate(initialProgram))
        environments = self.candidateEnvironments(initialProgram)
        e,l = self.sample(problem, environments)
        try:
            return self.applyChange(initialProgram, e, l)
        except: return initialProgram
        

class PolicyTrainingExample():
    def __init__(self, problem, target, environments):
        self.problem, self.target, self.environments = problem, target, environments
    def __str__(self):
        return "PolicyTrainingExample(problem = %s, target = %s, environments = %s)"%(self.problem, self.target, self.environments)

