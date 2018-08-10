import os
import sys

def eprint(s):
    print "[DEMO] %s"%s

def execute(k,n):
    print "[EXECUTING] %s:"%n
    print "\t%s"%k
    os.system(k + " 2>&1 | awk '{print \" [%s] \" $0}'"%n)

def getIndex(f):
    import filecmp
    try: return int(f)
    except:
        for j in xrange(100):
            if filecmp.cmp(f,"drawings/expert-%d.png"%j):
                return j
    return None
            
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "TikZ demo ^_^")
    parser.add_argument("image",type=str)
    parser.add_argument("task",
                        choices=["parse",
                                 "synthesize",
                                 "everything"],
                        default="everything",
                        type=str)
    parser.add_argument("--particles","-p",type=int,
                        default=100)
    parser.add_argument("--synthesizeFrom",
                        type=int,
                        default=None)
    parser.add_argument("--programOutputDirectory",
                        type=str,
                        default=None)
    parser.add_argument("--extra",
                        type=str,
                        default=None)
    parser.add_argument("--timeout",
                        type=int,
                        help="timeout for the synthesizer measured in seconds",
                        default=None)
    arguments = parser.parse_args()
    
    n = getIndex(arguments.image)
    imageFilename = "drawings/expert-%d.png"%n
    parseFileName = imageFilename[:-4] + "-parses"
    eprint("Demoing %s"%imageFilename)
    if arguments.task in ["parse","everything"]:
        eprint("Clearing parse directory: %s"%parseFileName)
        execute("rm %s/*.png %s/particle*.p"%(parseFileName,parseFileName),
                "RM")
        execute("python recognitionModel.py test  -t %s -b %d -l 0 --proposalCoefficient 1 --parentCoefficient --distanceCoefficient 5 --distance --mistakePenalty 10 --attention 16 --noisy --quiet "%
              (imageFilename,
               arguments.particles),
            "NEURALNET")
        eprint(" [+] Parsed image -- see %s/*.png"%parseFileName)

    if arguments.task in ["synthesize","everything"]:
        if arguments.synthesizeFrom is None:
            eprint("\tLaunching program synthesizer (ground truth parse)")
            synthesisTarget = str(n)
        else:
            eprint("\tLaunching program synthesizer (%d top-ranked empirical parse)"%(arguments.synthesizeFrom+1))
            synthesisTarget = parseFileName + "/particle%d.p"%arguments.synthesizeFrom

        if arguments.extra is None:
            extra = ""
        else:
            extra = "--extrapolate %s"%arguments.extra
            eprint("\tExtrapolating into %s"%arguments.extra)

        if arguments.timeout is None:
            timeout = ""
        else:
            timeout = "--timeout %d"%arguments.timeout

        if arguments.programOutputDirectory is None:
            outputDirectory = ""
        else:
            outputDirectory = "--programOutputDirectory %s"%arguments.programOutputDirectory

        execute("python synthesisPolicy.py %s %s %s -f basic --folds 1 --regularize 0.1 --load --evaluate %s "%(outputDirectory, extra,timeout,synthesisTarget),
                "SYNTHESIZER")
