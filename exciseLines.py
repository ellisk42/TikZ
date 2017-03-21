import os
import sys
from render import render

def excise(content):
    blocks = exciseTikZ(content)
    lines = []
    for b in blocks:
        lines += [ l+";" for l in b.split(";") if l ]
    return set(lines)

def exciseTikZ(content):
    blocks = []
    TikZ = False
    for line in content.splitlines():
        if "begin{tikzpicture}" in line:
            TikZ = True
            blocks.append("")
        elif "end{tikzpicture}" in line:
            TikZ = False
        elif TikZ:
            blocks[-1] += "\n"+line
    return blocks

def getLinesFromDirectory(directory):
    lines = set([])
    for f in os.listdir(directory):
        if f.endswith(".tex"):
            with open(directory+"/"+f) as h:
                lines |= excise(h.read())
    return lines

if __name__ == "__main__":
    lines = getLinesFromDirectory("TikZCrawl/data")
    j = 0
    for l in lines:
        render(l,"TikZCrawl/lines/%d.png"% j, None) # no canvas
        with open("TikZCrawl/lines/%d.tex"%j, 'w') as handle:
            handle.write(l + "\n")
        j += 1
        # every one percent
        if j%int(len(lines)/100.0) == 0:
            print "%d/%d" % (j,len(lines))
