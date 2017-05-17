from recognitionModel import RecognitionModel
from utilities import loadImage,removeBorder

import numpy as np
from sklearn.decomposition import PCA,NMF
from sklearn import preprocessing
from sklearn.manifold import MDS
import matplotlib.pyplot as plot
import matplotlib.image as image

class DummyArguments():
    def __init__(self):
        self.noisy = True
        self.distance = True
        self.architecture = "original"
        self.dropout = False
        self.learningRate = 1
        self.showParticles = False
        
def learnedDistanceMatrix(images):
    worker = RecognitionModel(DummyArguments())
    worker.loadDistanceCheckpoint("checkpoints/distance.checkpoint")

    jobs = [(x,y) for j,x in enumerate(images)
            for k,y in enumerate(images)
            if j != k ]
    distances = []
    while jobs != []:
        batch = jobs[:100]
        jobs = jobs[100:]

        print "%d jobs remaining"%len(jobs)

        result = worker.learnedDistances(np.array([x for x,_ in batch ]),
                                         np.array([x for _,x in batch ]))
        for j in range(len(batch)):
            distances.append(result[j,1] + result[j,0])

    matrix = np.zeros((len(images),len(images)))
    indexes = [(x,y) for x in range(len(images))
               for y in range(len(images))
               if x != y]
    for (x,y),d in zip(indexes, distances):
        matrix[x,y] += d

    matrix = matrix + matrix.T
    print "MATRIX:"
    print matrix.tolist()
    return matrix


def analyzeFeatures(featureMaps):
    # collect together a whole of the different names for features
    featureNames = list(set([ k for f in featureMaps.values() for k in f ]))

    imageNames = featureMaps.keys()

    # Convert feature maps into vectors
    featureVectors = [ [ featureMaps[k].get(name,0) for name in featureNames ]
                       for k in imageNames ]

    print "Feature vectors:"
    for j,n in enumerate(imageNames):
        print n
        print featureMaps[n]
        print featureVectors[j]

    for algorithm in range(3):
        if algorithm == 0:
            learner = PCA()
            transformedFeatures = learner.fit_transform(preprocessing.scale(np.array(featureVectors)))
            print learner.explained_variance_ratio_
        if algorithm == 1:
            learner = NMF(2)
            transformedFeatures = learner.fit_transform(preprocessing.scale(np.array(featureVectors),
                                                                            with_mean = False))
        if algorithm == 2:
            distances = learnedDistanceMatrix(map(loadImage,imageNames))
            learner = MDS(dissimilarity = 'precomputed')
            transformedFeatures = learner.fit_transform(distances)
            

        print transformedFeatures
        maximumExtent = max([transformedFeatures[:,0].max() - transformedFeatures[:,0].min(),
                             transformedFeatures[:,1].max() - transformedFeatures[:,1].min()])
        print maximumExtent

        if algorithm < 2:
            print learner.components_
            for dimension in range(2):
                coefficients = learner.components_[dimension]
                print "Dimension %d:"%(dimension+1)
                for j,n in enumerate(featureNames):
                    print n,'\t',learner.components_[dimension,j]
                print 

        
        f,a = plot.subplots()

        for j, imageName in enumerate(imageNames):
            i = 1 - image.imread(imageName)
            i = 1 - removeBorder(i)
            x = transformedFeatures[j,0]
            y = transformedFeatures[j,1]
            w = 0.05*maximumExtent
            a.imshow(i, aspect = 'auto',
                     extent = (x - w,x + w,
                               y - w,y + w),
                     zorder = -1,
                     cmap = plot.get_cmap('Greys'))

        a.scatter(transformedFeatures[:,0],
                  transformedFeatures[:,1])
        plot.show()


