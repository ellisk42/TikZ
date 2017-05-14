
class Architecture():
    def __init__(self, squareFilters,
                 rectangularFilters,
                 numberOfFilters,
                 kernelSizes,
                 poolSizes,
                 poolStrides):
        self.poolStrides = poolStrides
        self.squareFilters = squareFilters
        self.rectangularFilters = rectangularFilters
        self.numberOfFilters = numberOfFilters
        self.kernelSizes = kernelSizes
        self.poolSizes = poolSizes

architectures = {}
architectures["original"] = Architecture(12,4,
                                         numberOfFilters = [10],
                                         kernelSizes = [8],
                                         poolSizes = [8,4],
                                         poolStrides = [4,4])
architectures["v1"] = Architecture(20,2,
                                   numberOfFilters = [10],
                                   kernelSizes = [8],
                                   poolSizes = [8,4],
                                   poolStrides = [4,4])
architectures["v2"] = Architecture(20,2,
                                   numberOfFilters = [8,8],
                                   kernelSizes = [4,2],
                                   poolSizes = [2,4,2],
                                   poolStrides = [2,4,2])
