from language import *

def introduceNewVariable(program):
    def collectType(t,p):
        if isinstance(p,t): return [p]
        return [ pp for k in p.children() for pp in collectType(t,k) ]

    constantNumbers = set([ n.n for n in collectType(Number,program) ])

    if isinstance(program,Sequence):
        programLines = program
    else:
        programLines = Sequence([program])
        
    return [ Sequence([DefineConstant(Number(n))] + programLines.substitute(Number(n),Variable(0)).lines)
             for n in constantNumbers ]
        
        

if __name__ == '__main__':
    programs = introduceNewVariable(Sequence([Circle(AbsolutePoint(Number(1),Number(9)),Number(1)),
                                              Circle(AbsolutePoint(Number(1),Number(7)),Number(1))]))
    for p in programs:
        print p
    
