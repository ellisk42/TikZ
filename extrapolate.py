from utilities import *

def proposeExtrapolations(programs, N=30):
    trace = programs[0].explode().convertToSequence().removeDuplicates()
    print "original trace:"
    print trace
    originalUndesirability = trace.undesirabilityVector()
    print "original undesirability",originalUndesirability

    extrapolations = []

    extrapolationGenerators = [ program.explode().extrapolations() for program in programs ]
    for e in interleaveGenerators(extrapolationGenerators):
        t = e.convertToSequence().removeDuplicates()
        newUndesirability = t.undesirabilityVector()
        if (newUndesirability > originalUndesirability).sum() > 0: continue
        if t.canonicalTranslation() == trace.canonicalTranslation(): continue
        if any([t.canonicalTranslation() == o.canonicalTranslation() for o in extrapolations ]): continue
        extrapolations.append(t)

        if len(extrapolations) > N:
            break

    return extrapolations
