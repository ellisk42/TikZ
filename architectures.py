import tensorflow as tf


class Architecture():
    def __init__(self,
                 inputSize,
                 squareFilters,
                 rectangularFilters,
                 numberOfFilters,
                 kernelSizes,
                 poolSizes,
                 poolStrides):
        self.inputSize = inputSize
        self.poolStrides = poolStrides
        self.squareFilters = squareFilters
        self.rectangularFilters = rectangularFilters
        self.numberOfFilters = numberOfFilters
        self.kernelSizes = kernelSizes
        self.poolSizes = poolSizes

    def makeModel(self,imageInput):
        if imageInput.shape[1] != self.inputSize:
            imageInput = tf.image.resize(imageInput, [self.inputSize]*2, method=tf.image.ResizeMethod.BILINEAR)
        print("here")
        print(imageInput)
        print(self.rectangularFilters)
        print(self.kernelSizes[0])
        print(tf.nn.relu)
        print("/here")
        horizontalKernels = tf.compat.v1.layers.conv2d(inputs = imageInput,
                                             filters = self.rectangularFilters,
                                             kernel_size = [self.kernelSizes[0]*2,
                                                            self.kernelSizes[0]/2],
                                             padding = "same",
                                             activation = tf.nn.relu,
                                             strides = 1)
        verticalKernels = tf.compat.v1.layers.conv2d(inputs = imageInput,
                                             filters = self.rectangularFilters,
                                             kernel_size = [self.kernelSizes[0]/2,
                                                            self.kernelSizes[0]*2],
                                             padding = "same",
                                             activation = tf.nn.relu,
                                             strides = 1)
        squareKernels = tf.compat.v1.layers.conv2d(inputs = imageInput,
                                             filters = self.squareFilters,
                                             kernel_size = [self.kernelSizes[0],
                                                            self.kernelSizes[0]],
                                             padding = "same",
                                             activation = tf.nn.relu,
                                             strides = 1)
        c1 = tf.concat([horizontalKernels,verticalKernels,squareKernels], axis = 3)
        c1 = tf.compat.v1.layers.max_pooling2d(inputs = c1,
                                     pool_size = self.poolSizes[0],
                                     strides = self.poolStrides[0],
                                     padding = "same")

        numberOfFilters = self.numberOfFilters
        kernelSizes = self.kernelSizes[1:]
        poolSizes = self.poolSizes[1:]
        poolStrides = self.poolStrides[1:]
        nextInput = c1
        for filterCount,kernelSize,poolSize,poolStride in zip(numberOfFilters,kernelSizes,poolSizes,poolStrides):
            c1 = tf.compat.v1.layers.conv2d(inputs = nextInput,
                                  filters = filterCount,
                                  kernel_size = [kernelSize,kernelSize],
                                  padding = "same",
                                  activation = tf.nn.relu,
                                  strides = 1)
            c1 = tf.compat.v1.layers.max_pooling2d(inputs = c1,
                                         pool_size = poolSize,
                                         strides = poolStride,
                                         padding = "same")
            nextInput = c1
        return nextInput

architectures = {}
architectures["original"] = Architecture(256,
                                         12,4,
                                         numberOfFilters = [10],
                                         kernelSizes = [8,8],
                                         poolSizes = [8,4],
                                         poolStrides = [4,4])
architectures["v1"] = Architecture(256,
                                   20,2,
                                   numberOfFilters = [10],
                                   kernelSizes = [8,8],
                                   poolSizes = [8,4],
                                   poolStrides = [4,4])
architectures["v2"] = Architecture(256,
                                   20,2,
                                   numberOfFilters = [8,8],
                                   kernelSizes = [8,4,2],
                                   poolSizes = [2,4,2],
                                   poolStrides = [2,4,2])

architectures['v3'] = Architecture(128,
                                   20,2,
                                   numberOfFilters = [10],
                                   kernelSizes = [4,4],
                                   poolSizes = [4,4],
                                   poolStrides = [2,4])
architectures['v4'] = Architecture(128,
                                   32,2,
                                   numberOfFilters = [10],
                                   kernelSizes = [4,4],
                                   poolSizes = [4,4],
                                   poolStrides = [2,4])
architectures['v5'] = Architecture(256,
                                   32,2,
                                   numberOfFilters = [10],
                                   kernelSizes = [8,8],
                                   poolSizes = [8,4],
                                   poolStrides = [4,4])
architectures['v5'] = Architecture(256,
                                   32,2,
                                   numberOfFilters = [15],
                                   kernelSizes = [8,8],
                                   poolSizes = [8,4],
                                   poolStrides = [4,4])
# v1 was a little bit better
architectures["original"] = architectures["v1"]
