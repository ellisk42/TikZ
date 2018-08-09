import os
import sys

def execute(k,n):
    print "[EXECUTING] %s:"%n
    print "\t%s"%k
    os.system(k + " 2>&1 | awk '{print \" [%s] \" $0}'"%n)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "TikZ demo ^_^")
    parser.add_argument("index",type=int)
    parser.add_argument("--particles","-p",type=int,
                        default=100)
    arguments = parser.parse_args()
    
    n = arguments.index
    print "Demoing drawings/expert-%d.png"%n
    execute("python recognitionModel.py test  -t drawings/expert-%d.png -b %d -l 0 --proposalCoefficient 1 --parentCoefficient --distanceCoefficient 5 --distance --mistakePenalty 10 --attention 16 --noisy --quiet "%
              (n,
               arguments.particles),
            "NEURALNET")

    print " [+] Parsed image."

    print "\n\tLaunching program synthesizer (ground truth parse)"

    execute("python synthesisPolicy.py --extrapolate /tmp/extra_%d_gt.png -f basic --folds 1 --regularize 0.1 --load --evaluate %d "%(n,n),
            "SYNTHESIZER")

    print " [+] Synthesized from ground truth parse."

    print "\n\tLaunching program synthesizer (top-ranked empirical parse)"

    execute("python synthesisPolicy.py --extrapolate /tmp/extra_%d_empirical.png -f basic --folds 1 --regularize 0.1 --load --evaluate drawings/expert-%d-parses/particle0.p"%(n,n),
            "SYNTHESIZER")

    print " [+] Synthesized from top-ranked empirical parse."
