from DSL import *

from dispatch import dispatch

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
        if GPU: t = t.cuda()
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
        self.circleEncoder = DataEncoder(2, H)

        self.environmentScore = nn.Linear(H,1)
        
        

    def encodeProblem(self, s):
        return sum(\
                self.circleEncoder(Variable(t.cuda() if GPU else t)) \
                   for c in s\
                   for t in [torch.from_numpy(np.array([c.center.x, c.center.y])).float()])

    def encodeEnvironment(self, environment, seed):
        return self.lineEncoder.encoding(["START"] + environment, seed)

    def environmentLogLikelihoods(self, environments, problem):
        environmentEncodings = [ self.encodeEnvironment(e, problem)
                                 for e in environments ] 
        environmentScores = [ self.environmentScore(e)
                              for e in environmentEncodings ]
        return torch.nn.functional.log_softmax(torch.cat(environmentScores).view(-1))

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

class GraphicsSearchPolicy(SearchPolicy):
    def __init__(self):
        LEXICON = ["START","END",
                   "circle",
                   "rectangle",
                   "line","arrow = True","arrow = False","solid = True","solid = False",
                   "for",
                   "reflect","x","y",
                   "i","j","None"] + map(str,range(-5,20))
        super(GraphicsSearchPolicy,self).__init__(LEXICON)

    def candidateEnvironments(self, program):
        return list(e for e in candidateEnvironments(program))

    def applyChange(self, program, environment, line):
        if isinstance(program, Loop):
            return Loop(body = self.applyChange(program.body, environment, line),
                        v = program.v,
                        bound = program.bound)
        if isinstance(program, Reflection):
            return Reflection(body = self.applyChange(program.body, environment, line),
                              axis = program.axis,
                              coordinate = program.coordinate)

        assert isinstance(program, Block)
        
        if environment == []:
            return Block([self.parseLine(line)] + program.items)
        # Figure out what it is that's being indexed into
        for j,l in enumerate(program.items):
            s = serializeLine(l)
            if s == environment[:len(s)]:
                lp = self.applyChange(l, environment[len(s):], line)
                newItems = list(program.items)
                newItems[j] = lp
                return Block(newItems)
        raise Exception('Environment indexes nonexistent context')
    
    def Oracle(self, program): return list(Oracle(program))

    def evaluate(self, program): return program.evaluate(Environment([]))
    def residual(self, goal, current):
        #assert len(current - goal) == 0
        return goal - current

    def parseLine(self, l):
        def get(l):
            n = l[0]
            del l[0]
            return n
        def finish(l):
            if l != []: raise Exception('Extra symbols in line')
        def parseLinear(l):
            b = int(get(l))
            x = get(l)
            m = int(get(l))
            if x == 'None': x = None
            return LinearExpression(m,x,b)
        k = get(l)
        if k == 'circle':
            x = parseLinear(l)
            y = parseLinear(l)
            finish(l)
            return Primitive(k,x,y)
        if k == 'for':
            v = get(l)
            b = parseLinear(l)
            finish(l)
            return Loop(v = v, bound = b, body = Block([]))
        if k == 'reflect':
            a = get(l)
            c = int(get(l))
            finish(l)
            return Reflection(body = Block([]), axis = a, coordinate = c)
        raise Exception('parsing line '+k)
            

    
        
        
@dispatch(Block)
def Oracle(b):
    for j,x in enumerate(b.items):
        serialized = serializeLine(x)
        yield Block(b.items[:j]), [], serialized
        for program, environment, line in Oracle(x):
            yield Block(b.items[:j] + [program]), serialized + environment, line
@dispatch(Primitive)
def Oracle(p):
    return
    yield
@dispatch(Loop)
def Oracle(l):
    for program, environment, line in Oracle(l.body):
        yield Loop(v = l.v, bound = l.bound, body = program), environment, line
@dispatch(Reflection)
def Oracle(l):
    for program, environment, line in Oracle(l.body):
        yield Reflection(axis = l.axis, coordinate = l.coordinate, body = program), environment, line
    


@dispatch(Loop)
def serializeLine(l):
    return ["for",l.v] + serializeLine(l.bound)
@dispatch(Reflection)
def serializeLine(r):
    return ["reflect",r.axis,str(r.coordinate)]
@dispatch(LinearExpression)
def serializeLine(e):
    return [str(e.b),str(e.x),str(e.m)]
@dispatch(Primitive)
def serializeLine(p):
    s = [p.k]
    for a in p.arguments[:4]:
        s += serializeLine(a)
    if p.k == 'line':
        s += ["arrow = True" if "True" in p.arguments[4] else "arrow = False" ]
        s += ["solid = True" if "True" in p.arguments[5] else "solid = False"]
    return s

@dispatch(Circle)
def serializeObservation(c):
    return ["circle",str(c.center.x),str(c.center.y)]
@dispatch(Rectangle)
def serializeObservation(c):
    return ["rectangle",str(c.p1.x),str(c.p1.y),str(c.p2.x),str(c.p2.y)]

@dispatch(Block)
def candidateEnvironments(b):
    yield []
    for x in b.items:
        for e in candidateEnvironments(x):
            yield e
@dispatch(Primitive)
def candidateEnvironments(_):
    return
    yield 
@dispatch(Loop)
def candidateEnvironments(l):
    this = serializeLine(l)
    for e in candidateEnvironments(l.body):
        yield this + e
@dispatch(Reflection)
def candidateEnvironments(r):
    this = serializeLine(l)
    for e in candidateEnvironments(l.body):
        yield this + e

def simpleSceneSample():
    def isolatedCircle():
        x = random.choice(range(1,16))
        y = random.choice(range(1,16))
        return Primitive('circle', LinearExpression(0,None,x), LinearExpression(0,None,y))

    primitives = [isolatedCircle() for _ in range(random.choice(range(2))) ]
    loopIterations = random.choice([3,4])

    while True:
        bx = random.choice(range(1,16))
        mx = random.choice(range(-5,6))
        if all([x > 0 and x < 16 for j in range(loopIterations) for x in [mx*j + bx] ]): break
    while True:
        by = random.choice(range(1,16))
        my = random.choice(range(-5,6))
        if my == 0 and mx == 0: continue
        
        if all([y > 0 and y < 16 for j in range(loopIterations) for y in [my*j + by] ]): break



    l = Loop(v = 'j',bound = LinearExpression(0,None,loopIterations),
             body = Block([Primitive('circle',
                                     LinearExpression(mx,'j' if mx else None,bx),
                                     LinearExpression(my,'j' if my else None,by))]))
    return Block([l] + primitives)
    
    
if __name__ == "__main__":
    p = GraphicsSearchPolicy()
    if GPU:
        print "Using the GPU"
        p.cuda()

    o = optimization.Adam(p.parameters(), lr = 0.001)
    criteria = nn.CrossEntropyLoss()
    
    step = 0
    while True:
        step += 1

        
        program = simpleSceneSample()
        scene = set(program.convertToSequence().lines)

        examples = p.makeOracleExamples(program, scene)
        

        losses = []
        for example in examples:
            o.zero_grad()
            loss = p.loss(example, criteria)
            loss.backward()
            o.step()
            losses.append(loss.data[0])

        if step%100 == 0:
            torch.save(p.state_dict(),'checkpoints/neuralSearch.p')
            print step,'\t',sum(losses)/len(losses)
            losses = []
            print scene
            print program.pretty()
            p0 = Block([])
            for _ in range(5):
                p0 = p.sampleOneStep(scene, p0)
                print p0
                if len(scene - p.evaluate(p0)) == 0:
                    print "Nothing left to explain."
                    break
                
