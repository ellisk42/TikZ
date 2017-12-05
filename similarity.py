from recognitionModel import RecognitionModel
from utilities import loadImage,removeBorder,showImage,saveMatrixAsImage


import random
import numpy as np

class DummyArguments():
    def __init__(self):
        self.noisy = True
        self.distance = True
        self.architecture = "original"
        self.dropout = False
        self.learningRate = 1
        self.showParticles = False
        
def learnedDistanceMatrix(images):
    from calculate_distances import distanceMatrix
    return np.array(distanceMatrix)
    # worker = RecognitionModel(DummyArguments())
    # worker.loadDistanceCheckpoint("checkpoints/distance.checkpoint")

    # jobs = [(x,y) for j,x in enumerate(images)
    #         for k,y in enumerate(images)
    #         if j != k ]
    # distances = []
    # while jobs != []:
    #     batch = jobs[:100]
    #     jobs = jobs[100:]

    #     print "%d jobs remaining"%len(jobs)

    #     result = worker.learnedDistances(np.array([x for x,_ in batch ]),
    #                                      np.array([x for _,x in batch ]))
    #     for j in range(len(batch)):
    #         distances.append(result[j,1] + result[j,0])

    # matrix = np.zeros((len(images),len(images)))
    # indexes = [(x,y) for x in range(len(images))
    #            for y in range(len(images))
    #            if x != y]
    # for (x,y),d in zip(indexes, distances):
    #     matrix[x,y] += d

    # matrix = matrix + matrix.T
    # print "MATRIX:"
    # print matrix.tolist()
    # return matrix

if __name__ == '__main__':
    worker = RecognitionModel(DummyArguments())
    worker.loadDistanceCheckpoint("checkpoints/distance.checkpoint")

    imageNames = ['drawings/expert-%d.png'%j
                  for j in range(100) ]
    images = [ loadImage(n)
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
    from sklearn.decomposition import PCA,NMF
    from sklearn import preprocessing
    from sklearn.manifold import MDS
    import matplotlib.pyplot as plot
    import matplotlib.image as image

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

    # Figure out things that are close / far as measured by different metrics
    percentile = 80
    imageDistances = learnedDistanceMatrix(None)
    numberOfPrograms = len(featureVectors)
    programDistances = np.zeros((numberOfPrograms,numberOfPrograms))
    featureVectors = preprocessing.scale(np.array(featureVectors))
    for j in range(numberOfPrograms):
        for k in range(numberOfPrograms):
            programDistances[j,k] = ((featureVectors[j,:] - featureVectors[k,:])*(featureVectors[j,:] - featureVectors[k,:])).sum()
    smallDistance = np.percentile(programDistances,100 - percentile)
    bigDistance = np.percentile(programDistances,percentile)
    closePrograms = set([(n1,n2) for j,n1 in enumerate(imageNames) for k,n2 in enumerate(imageNames)
                     if n1 < n2 and programDistances[j,k] < smallDistance])
    farPrograms = set([(n1,n2) for j,n1 in enumerate(imageNames) for k,n2 in enumerate(imageNames)
                   if n1 < n2 and programDistances[j,k] > bigDistance])
    smallDistance = np.percentile(imageDistances,100 - percentile)
    bigDistance = np.percentile(imageDistances,percentile)
    imageNames = ["drawings/expert-%d.png"%j for j in range(100) ]
    closeImages = set([(n1,n2) for j,n1 in enumerate(imageNames) for k,n2 in enumerate(imageNames)
                       if n1 < n2 and imageDistances[j,k] < smallDistance])
    farImages = set([(n1,n2) for j,n1 in enumerate(imageNames) for k,n2 in enumerate(imageNames)
                     if n1 < n2 and imageDistances[j,k] > bigDistance])
    programOptions = [(closePrograms,'close in program space'),(farPrograms,'distant in program space')]
    imageOptions = [(closeImages,'close in image space'),(farImages,'far in image space')]
    for programSet,programName in programOptions:
        for imageSet,imageName in imageOptions:
            overlap = programSet&imageSet
            print programName,'&',imageName,'have overlap',len(overlap)
            overlap = list(sorted(list(overlap)))
            indices = np.random.choice(range(len(overlap)),size = min(100,len(overlap)),replace = False)
            overlap = [overlap[j] for j in indices ]
            
            matrix = 1 - np.concatenate([ np.concatenate((loadImage(n1),loadImage(n2)), axis = 0) for n1,n2 in overlap ],axis = 1)
            saveMatrixAsImage(matrix*255,"similarity/%s%s.png"%(programName,imageName))            



    assert False
    

    for algorithm in [0,1,2]:
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
        w = 0.1*maximumExtent
        
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
                d = abs(transformedFeatures[j,:2] - transformedFeatures[k,:2])
                if d[0] < 2*w and d[1] < 2*w:
                    overlapping += 1
            
            showProbability.append(1.0/(1 + overlapping*0.3))

        for index in range(50):
            f,a = plot.subplots()
            imageIsShown = [random.random() < sp for sp in showProbability ]
            coolImages = [38,39,12,26,46,47,76,71,75,80,89]
            for coolImage in coolImages:
                coolImage = 'drawings/expert-%d.png'%coolImage
                for j, imageName in enumerate(imageNames):
                    if imageName == coolImage:
                        imageIsShown[j] = True
                        break
                    
                
            imageCoordinates = [ transformedFeatures[j,:2] for j in range(len(imageNames)) ]
            imageCoordinates = diffuseImagesOutwards(imageCoordinates,w,imageIsShown)

            for j, imageName in enumerate(imageNames):
                if not imageIsShown[j]: continue
                
                i = 1 - image.imread(imageName)
                i = 1 - removeBorder(i)
                x = imageCoordinates[j][0]
                y = imageCoordinates[j][1]

                a.imshow(i, aspect = 'auto',
                         extent = (x - w,x + w,
                                   y - w,y + w),
                         zorder = -1,
                         cmap = plot.get_cmap('Greys'))
                a.arrow(x, y,
                        transformedFeatures[j,0] - x, transformedFeatures[j,1] - y,
                        head_width=0.04, head_length=0.1, fc='k', ec='k',
                        zorder = -2)

            a.scatter(transformedFeatures[:,0],
                      transformedFeatures[:,1])
            a.get_yaxis().set_visible(False)
            a.get_xaxis().set_visible(False)

            n = ['PCA','NMF','MDS'][algorithm]
            plot.savefig('%s/%s%d.png'%(n,n,index),bbox_inches = 'tight')
#            plot.show()



def diffuseImagesOutwards(coordinates, w, mask):
    cp = coordinates
    for _ in range(10):

        forces = []
        for j,p1 in enumerate(cp):
            F = np.array([0.0,0.0])
            for k,p2 in enumerate(cp):
                if (not mask[k]) or (not mask[j]): continue
                if j == k: continue
                distance = ((p1 - p2)*(p1 - p2)).sum()
                if distance == 0: continue
                difference = abs(p1 - p2)
                if difference[0] > 2*w or difference[1] > 2*w: continue
                
                F += (p1 - p2)/distance
            forces.append(F)
        cp = [ c + f for c,f in zip(cp,forces) ]
        allowedDistance = 1.3
        cp = [ np.array([ min(original[0] + allowedDistance*w, max(original[0] - allowedDistance*w, c[0])),
                          min(original[1] + allowedDistance*w, max(original[1] - allowedDistance*w, c[1]))])
               for c,original in zip(cp,coordinates) ]

    return cp
            
