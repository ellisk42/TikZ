

class GeneticAlgorithm():
    def __init__(self): pass

    def randomIndividual(self):
        raise Exception('randomIndividual: not implemented')

    def mutate(self, candidate):
        raise Exception('mutate: not implemented')

    def fitness(self, candidate):
        raise Exception('fitness: not implemented')

    def mapFitness(self, candidates):
        return [self.fitness(k) for k in candidates ]

    def beam(self, generations, beamSize, branchingFactor):
        population = [ self.randomIndividual() for _ in range(beamSize) ]
        bestFitness = float('-inf')
        bestIndividual = None

        bestHistory = []

        for g in range(generations):
            print("Generation",g)

            print("Expanding population via mutation")
            expandedPopulation = [ child
                                   for parent in population
                                   for child in [ self.mutate(parent) for _ in range(branchingFactor) ] ]
            print("Computing fitness")
            expandedFitness = self.mapFitness(expandedPopulation)
            print("Done with fitness")
            expandedPopulation = set(zip(expandedFitness, expandedPopulation))
            
            population = sorted(list(expandedPopulation))
            population.reverse()
            if len(population) > beamSize:
                population = population[:beamSize]

            if population[0][0] > bestFitness:
                bestFitness = population[0][0]
                bestIndividual = population[0][1]
                print("Found a new best individual:")
                print(bestIndividual)
                print("Fitness:",bestFitness)
                bestHistory.append(bestIndividual)
            population = [ individual[1] for individual in population ]

            if bestFitness > -1.0:
                print("Terminating early")
                break
            
        return bestIndividual, bestHistory
            
            
