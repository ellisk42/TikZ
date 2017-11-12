

To generate synthetic training data for the neural network:
```
python makeSyntheticData.py 100000
```
which will generate 100000 training examples in to the file `syntheticTrainingData.tar`

To train the neural networks:
```
python recognitionModel.py train --noisy  --attention 16 # trains the proposal distribution
python recognitionModel.py train --noisy  --distance # trains the distance metric
```

To run the neural network on all of the images in a directory called `drawings/`, with 1000 particles, do:
```
python recognitionModel.py test  -t drawings -b  1000 -l 0 --proposalCoefficient 1 --parentCoefficient --distanceCoefficient 5 --distance --mistakePenalty 10 --attention 16 --noisy --quiet
```

To use the program synthesizer you will need `sketch`:
```
https://people.csail.mit.edu/asolar/
```
I used sketch 1.7.5.

To run the program synthesizer on the 38th drawing (found in `drawings/expert-38.png`), do:
```
python synthesizer.py -f 38 # to pass the entire problem all at once the sketch
python synthesizer.py -f 38 --incremental  # to break the problem up into pieces and pass each piece to sketch
```

To synthesize all of the programs for all of the drawings in every way possible and distribute the work across twenty CPUs, do:
```
python synthesizer.py --makePolicyTrainingData --cores 20
```

To view some extrapolations, do:
```
python synthesizer.py -n policyTrainingData.p --view --extrapolate
```
which should place its outputs into `extrapolations/`.
