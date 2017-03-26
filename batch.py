import numpy as np

class BatchIterator():
    def __init__(self, batchSize, tensors, stringProcessor = None):
        for t in tensors: assert t.shape[0] == tensors[0].shape[0]
        
        # side-by-side shuffle of the data
        permutation = np.random.permutation(range(tensors[0].shape[0]))
        self.tensors = [ np.array([ t[p,...] for p in permutation ]) for t in tensors ]
        self.batchSize = batchSize
        
        self.startingIndex = 0
        self.trainingSetSize = tensors[0].shape[0]

        self.process = stringProcessor

    def processTensor(self,t):
        if not isinstance(t[0],str): return t
        return np.array(map(self.process, list(t)))
    
    def next(self):
        endingIndex = self.startingIndex + self.batchSize
        if endingIndex > self.trainingSetSize:
            endingIndex = self.trainingSetSize
        batch = tuple([ self.processTensor(t[self.startingIndex:endingIndex,...]) for t in self.tensors ])
        self.startingIndex = endingIndex
        if self.startingIndex == self.trainingSetSize: self.startingIndex = 0
        return batch
