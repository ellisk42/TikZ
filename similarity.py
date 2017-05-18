from recognitionModel import RecognitionModel
from utilities import loadImage,removeBorder


import random
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
    # from calculate_distances import distanceMatrix
    # return np.array(distanceMatrix)
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

if __name__ == '__main__':
    worker = RecognitionModel(DummyArguments())
    worker.loadDistanceCheckpoint("checkpoints/distance.checkpoint")

    imageNames = ['drawings/expert-%d.png'%j
                  for j in range(100) ]
    images = [ loadImage('/om/user/ellisk/%s'%n)
               for n in imageNames ]
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

    for algorithm in [2]:
        if algorithm == 0:
            learner = PCA()
            transformedFeatures = learner.fit_transform(preprocessing.scale(np.array(featureVectors)))
            print learner.explained_variance_ratio_
        if algorithm == 1:
            learner = NMF(2)
            transformedFeatures = learner.fit_transform(preprocessing.scale(np.array(featureVectors),
                                                                            with_mean = False))
        if algorithm == 2:
            imageNames = ["drawings/expert-%d.png"%j for j in range(100) ]
            distances = learnedDistanceMatrix(map(loadImage,imageNames))
            learner = MDS(dissimilarity = 'precomputed')
            transformedFeatures = learner.fit_transform(distances)
            

        print transformedFeatures
        maximumExtent = max([transformedFeatures[:,0].max() - transformedFeatures[:,0].min(),
                             transformedFeatures[:,1].max() - transformedFeatures[:,1].min()])
        print maximumExtent
        w = 0.09*maximumExtent
        
        if algorithm < 2:
            print learner.components_
            for dimension in range(2):
                coefficients = learner.components_[dimension]
                print "Dimension %d:"%(dimension+1)
                for j,n in enumerate(featureNames):
                    print n,'\t',learner.components_[dimension,j]
                print 

        
        

        showProbability = []
        for j in range(len(imageNames)):
            overlapping = 0
            for k in range(len(imageNames)):
                if j == k: continue
                d = transformedFeatures[j,:] - transformedFeatures[k,:]
                d = d[0]*d[0] + d[1]*d[1]
                if d < w*w:
                    overlapping += 1
            
            showProbability.append(1.5/(1 + overlapping))

        for index in range(50):
            f,a = plot.subplots()
            for j, imageName in enumerate(imageNames):
                if random.random() > showProbability[j]: continue

                i = 1 - image.imread(imageName)
                i = 1 - removeBorder(i)
                x = transformedFeatures[j,0]
                y = transformedFeatures[j,1]

                a.imshow(i, aspect = 'auto',
                         extent = (x - w,x + w,
                                   y - w,y + w),
                         zorder = -1,
                         cmap = plot.get_cmap('Greys'))

            a.scatter(transformedFeatures[:,0],
                      transformedFeatures[:,1])
            a.get_yaxis().set_visible(False)
            a.get_xaxis().set_visible(False)

            n = ['PCA','NMF','MDS'][algorithm]
            plot.savefig('%s/%s%d.png'%(n,n,index),bbox_inches = 'tight')
#            plot.show()



