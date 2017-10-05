from synthesizer import *
from utilities import sampleLogMultinomial

import numpy as np


import tensorflow as tf

#def torchlse(#stuff):
    
# disgusting
canonicalJobOrdering = [(i,l,r,d) for i in  [True,False]
                        for l in  [True,False]
                        for r in [True,False]
                        for d in [1,2,3]]
def canonicalIndex(j): return canonicalJobOrdering.index((j.incremental,
                                                          j.canLoop,
                                                          j.canReflect,
                                                          j.maximumDepth))

class SynthesisPolicy():
    def __init__(self):
        self.inputDimensionality = len(SynthesisPolicy.featureExtractor(Sequence([])))
        self.outputDimensionality = 6
        self.B = 100

        self.features = tf.Placeholder(tf.float32,[None,self.inputDimensionality])

        self.incrementalPrediction = tf.layers.dense(self.features, 1,
                                                     activation = tf.nn.sigmoid)
        self.loopPrediction = tf.layers.dense(self.features, 1,
                                                     activation = tf.nn.sigmoid)
        self.reflectPrediction = tf.layers.dense(self.features, 1,
                                                 activation = tf.nn.sigmoid)
        self.depthPrediction = nn.layers.dense(self.features,3,
                                               activation = tf.nn.softmax)

        
        

    def expectedTime(self,results):
        jobLikelihood = {}
        for j in results:
            jobLikelihood[j] = (self.incrementalPrediction if j.incremental else 1 - self.)
                               
        

        

    @staticmethod
    def featureExtractor(sequence):
        return np.array([len([x for x in sequence.lines if isinstance(x,k) ])
                for k in [Line,Circle,Rectangle]])

    def rollout(self, sequence, results, session):
        f = SynthesisPolicy.featureExtractor(sequence)

        [i,l,r,d] = session.run([self.incrementalPrediction,
                                 self.loopPrediction,
                                 self.reflectPrediction,
                                 self.depthPrediction],
                                feed_dict = {self.features:f.reshape([-1,1])})

        jobLikelihood = {}
        for j,result in results.iteritems():        
            jobLikelihood[j] = l[0,int(j.canLoop)]*r[0,int(j.canReflect)]*i[0,int(j.incremental)]*d[0,int(j.maximumDepth - 1)]

        history = []
        TIMEOUT = 999
        minimumCost = min([ r.cost for r in results.values() if r.cost != None ] + [TIMEOUT])
        if minimumCost == TIMEOUT:
            print "TIMEOUT",sequence
            assert False

        time = 0
        while True:
            candidates = [ j
                           for j,_ in results.iteritems()
                           if not any([ o.subsumes(j) for o in history ])]
            job = candidates[np.random.multinomial(1,[ jobLogLikelihood[j].data for j in candidates ]).tolist.index(1)]
            sample = results[job]
            time += sample.time
            history.append(job)
            
            if sample.cost != None and sample.cost <= minimumCost + 1:
                return time
            
                
            
        

        # Sample a job
        i_ = torch.bernoulli(torch.exp(i.data))
        l_ = torch.bernoulli(torch.exp(l.data))
        r_ = torch.bernoulli(torch.exp(r.data))
        d_ = torch.multinomial(torch.exp(d.data),1) + 1
        
        
def loadPolicyData():
    with open('policyTrainingData.p','rb') as handle:
        results = pickle.load(handle)

    resultsArray = []

    for j in range(100):
        drawing = 'drawings/expert-%d.png'%j
        resultsArray.append(dict([ (r.job, r) for r in results if isinstance(r,SynthesisResult) and r.job.originalDrawing == drawing ]))
        print " [+] Got %d results for %s"%(len(resultsArray[-1]), drawing)

    return resultsArray

 
            

def evaluatePolicy(results, policy):
    jobs = results.keys()
    minimumCost = min([ r.cost for r in results.values() if r.cost != None ])
    scores = map(policy, jobs)
    orderedJobs = sorted(zip(scores, jobs), reverse = True)
    print map(lambda oj: str(snd(oj)),orderedJobs)
    events = []
    T = 0.0
    minimumCostSoFar = float('inf')
    for j, (score, job) in enumerate(orderedJobs):
        if any([ o.subsumes(job) for _,o in orderedJobs[:j] ]): continue

        T += results[job].time

        if results[job].cost == None: continue
        
        normalizedCost = minimumCost/float(results[job].cost)

        if normalizedCost < minimumCostSoFar:
            minimumCostSoFar = normalizedCost
            events.append((T,normalizedCost))
    return events

TIMEOUT = 10*60*60
def bestPossibleTime(results):
    minimumCost = min([ r.cost for r in results.values() if r.cost != None ] + [TIMEOUT])
    return math.log(min([ r.time for r in results.values() if r.cost != None and r.cost <= minimumCost + 1 ] + [TIMEOUT]))
def exactTime(results):
    return math.log(min([ r.time for j,r in results.iteritems()
                          if j.incremental == False and j.canLoop and j.canReflect and j.maximumDepth == 3] + [TIMEOUT]))
def incrementalTime(results):
    return math.log(min([ r.time for j,r in results.iteritems()
                          if j.incremental and j.canLoop and j.canReflect and j.maximumDepth == 3] + [TIMEOUT]))
        
if __name__ == '__main__':
    data = loadPolicyData()

    SynthesisPolicy().rollout(data[38].keys()[0].parse,
                              data[38])


    optimistic = map(bestPossibleTime,data)
    exact = map(exactTime,data)
    incremental = map(incrementalTime,data)

    print exact

    import matplotlib.pyplot as plot
    import numpy as np
    
    bins = np.linspace(0,20,40)
    for ys,l in [(exact,'exact'),(optimistic,'optimistic'),(incremental,'incremental')] :
        plot.hist(ys, bins, alpha = 0.3, label = l)
    plot.legend()
    plot.show()
    
    for j,r in data.iteritems():
        if r.cost  == None: continue

        print j
        print r.cost
        print int(r.time/60.0),'m'
        print
    print evaluatePolicy(data, lambda j: int(j.incremental)) #featureExtractor(j.parse))
