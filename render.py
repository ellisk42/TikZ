from utilities import showImage
import tempfile
import sys
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.animation as animation

def render(sources, showImage = False, yieldsPixels = False, canvas = (16,16), resolution = 256):
    # definitely do not try to render too much at once - I think this causes memory problems
    if len(sources) > 100:
        prefix = render(sources[:100], showImage, yieldsPixels, canvas, resolution)
        suffix = render(sources[100:], showImage, yieldsPixels, canvas, resolution)
        return prefix + suffix
    if canvas == None: canvas = ""
    else: canvas = '''
\draw[fill = white, white] (0,0) rectangle (%d,%d);
'''%(canvas[0],canvas[1])

    preamble = "\\begin{tikzpicture}"
    preamble += "[pencildraw/.style={black,decorate,decoration={random steps,segment length=4pt,amplitude=1pt}}]"
    preamble += "\n"
    sources = [preamble + canvas + "\n" + s + "\n\\end{tikzpicture}"
               for s in sources ]
    source = "\n\n\n".join(sources)
    source = '''
\\documentclass[convert={density=300,size=%dx%d,outext=.png},tikz]{standalone}
\\usetikzlibrary{decorations.pathmorphing}
\\usetikzlibrary{arrows.meta}
\\begin{document}
%s
\\end{document}
''' % (resolution, resolution, source)

    fd, temporaryName = tempfile.mkstemp(suffix = ".tex")

    with os.fdopen(fd, 'w') as new_file:
        new_file.write(source)
    os.system("cd /tmp; echo X|pdflatex -shell-escape %s 2> /dev/null > /dev/null" % temporaryName)

    temporaryPrefix = temporaryName[:-4]
    temporaryImages = [temporaryPrefix + ".png"]
    if len(sources) > 1:
        pattern = "%s-%0"+str(len(str(len(sources) - 1)))+"d.png"
        # for sketch2
        pattern = "%s-%d.png"
        temporaryImages = [pattern%(temporaryPrefix,j) for j in range(len(sources)) ]

    if showImage:
        for temporaryImage in temporaryImages:
            os.system("feh %s" % temporaryImage)

    returnValue = []
    if yieldsPixels:
        for temporaryImage in temporaryImages:
            im = Image.open(temporaryImage)
            (width, height) = im.size
            if width != resolution or height != resolution:
                im = im.resize((resolution,resolution))
            if im.mode == 'RGBA' or im.mode == '1':
                im = im.convert('L')
                scale = 255.0
            elif im.mode == 'I': # why God does this happen
                scale = 65535.0
            else:
                raise Exception('Unhandled image format:'+im.mode)
            greyscale_map = list(im.getdata())
            greyscale_map = np.array(greyscale_map)
            greyscale_map = greyscale_map.reshape((resolution, resolution))
            returnValue.append(greyscale_map/scale)


    os.system("rm %s*" % temporaryPrefix)
    if returnValue != []: return returnValue

def animateMatrices(matrices,outputFilename = None):
    fig = plot.figure() # make figure
    im = plot.imshow(matrices[0], cmap=plot.get_cmap('bone'), vmin=0.0, vmax=1.0)

    # function to update figure
    def updatefig(j):
        # set the data in the axesimage object
        im.set_array(matrices[j])
        # return the artists set
        return im,
    # kick off the animation
    ani = animation.FuncAnimation(fig, updatefig, frames=range(len(matrices)), 
                              interval=50, blit=True)
    if outputFilename != None:
        ani.save(outputFilename, dpi = 80,writer = 'imagemagick')
    plot.show()

if __name__ == "__main__":
    challenge = '''
    \\node[draw,circle,inner sep=0pt,minimum size = 2cm,line width = 0.1cm] at (3,5) {};
    \\node[pencildraw,draw,circle,inner sep=0pt,minimum size = 2cm,ultra thick] at (7,5) {};
    \\draw[pencildraw,line width = 0.1cm,dashed,-{>[scale = 1.5]}] (4,5) -- (6,5);
'''
    showImage(render([challenge],showImage = False,yieldsPixels = True)[0])
    # inputFile = sys.argv[1]
    # outputFile = sys.argv[2]
    # i = sys.stdin if inputFile == '-' else open(inputFile, "r")
    # source = i.read()
    # render(source, outputFile)
