from utilities import loadImage

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plot
import matplotlib.image as image


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
    
    learner = PCA()
    transformedFeatures = learner.fit_transform(np.array(featureVectors))
    print learner.explained_variance_ratio_


    print transformedFeatures
    f,a = plot.subplots()
    
    for j, imageName in enumerate(imageNames):
        i = image.imread(imageName)
        x = transformedFeatures[j,0]
        y = transformedFeatures[j,1]
        w = 0.2
        a.imshow(i, aspect = 'auto',
                 extent = (x - w,x + w,
                           y - w,y + w),
                 zorder = -1,
                 cmap = plot.get_cmap('Greys'))
#        break
    a.scatter(transformedFeatures[:,0],
              transformedFeatures[:,1])
    plot.show()
