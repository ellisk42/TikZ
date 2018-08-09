from utilities import *

def proposeExtrapolations(programs, N=30):
    trace = programs[0].explode().convertToSequence().removeDuplicates()
    originalUndesirability = trace.undesirabilityVector()

    extrapolations = []

    extrapolationGenerators = [ program.explode().extrapolations() for program in programs ]
    for e in interleaveGenerators(extrapolationGenerators):
        t = e.convertToSequence().removeDuplicates()
        newUndesirability = t.undesirabilityVector()
        badness =  (newUndesirability > originalUndesirability).sum() > 0
        if t.canonicalTranslation() == trace.canonicalTranslation(): continue
        if any([t.canonicalTranslation() == o.canonicalTranslation() for _,o in extrapolations ]): continue
        extrapolations.append((badness,t))

    extrapolations.sort(key=lambda bo: bo[0])
    return [o for _,o in extrapolations ][:N]

def exportExtrapolations(programs, fn, index=None):
    extrapolations = proposeExtrapolations(programs)
    framedExtrapolations = [ frameImageNicely(t.draw(adjustCanvasSize = True))
                             for t in extrapolations ]
    if index is not None:
        framedExtrapolations = [1 - frameImageNicely(loadImage(index))] + framedExtrapolations
    a = 255*makeImageArray(framedExtrapolations)
    saveMatrixAsImage(a,fn)

