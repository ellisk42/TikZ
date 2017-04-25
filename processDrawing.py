import sys
import numpy as np
from PIL import Image
from utilities import showImage,image2array
import os

def processPDF(name):
    os.system('convert -density 150 %s -quality 90 /tmp/output.png'%name)
    for j in range(100):
        _j = j
        p = '/tmp/output-%d.png'%j
        if not os.path.isfile(p): break

        x = Image.open(p).convert('L')
        (w,h) = x.size
        # only take the top half of the image
        x = x.crop((0,0,w,h/2))

        # flip it horizontally if the image is on the right
        (w,h) = x.size
        a = image2array(x)
        leftEnergy = a[:,0:w/2].sum()
        rightEnergy = a[:,w/2:].sum()
        flipHorizontally = rightEnergy > leftEnergy

        if flipHorizontally:
            x = x.transpose(Image.FLIP_LEFT_RIGHT)


        #showImage(image2array(x))

        # try to locate the upper left corner
        a = image2array(x)
        scanThreshold = 7
        initialY = None
        for j in range(10,a.shape[0]/2):
            if a[j,:].sum() > scanThreshold:
                initialY = j
                break
        initialX = None
        for j in range(10,a.shape[0]/2):
            if a[:,j].sum() > scanThreshold:
                initialX = j
                break

        # remove left and upper borders
        x = x.crop((initialX,initialY,
                    w,h))

        gridWidth = 38
        x = x.crop((0,0,
                    gridWidth*15,gridWidth*15))

        a = image2array(x)
        for j in range(15):
            a[:,j*gridWidth] = 1
            a[j*gridWidth,:] = 1
#        showImage(a)

        intendedGridWidth = 256/16
        rescaleFactor = float(intendedGridWidth)/float(gridWidth)
        x = x.resize((intendedGridWidth*15, intendedGridWidth*15),
                     Image.BILINEAR)
        
        # insert a buffer to the left and above which is one grid width
        a = image2array(x)
#        showImage(a)
        z = np.zeros((intendedGridWidth,intendedGridWidth*15))
        a = np.concatenate([z,a],axis = 0)
        z = np.zeros((intendedGridWidth*16,intendedGridWidth))
        a = np.concatenate([z,a],axis = 1)

        # adjust colors
        a = a*2
        a[a > 1.0] = 1.0
        a[a < 0.1] = 0

        # this verifies that indeed the grid lines have been completely removed
        # a = a*100
        # a[a > 1] = 1

        # showImage(a)
        # continue
    
    

        # negative, so that it is black on white
        a = 1.0 - a

        x = Image.fromarray(a*255).convert('L')
        if flipHorizontally:
            x = x.transpose(Image.FLIP_LEFT_RIGHT)

        x.save('drawings/scan-%d.png'%_j)
#        showImage(a)

def processDrawing(name, export = False):
    if 'pdf' in name:
        return processPDF(name)
    x = Image.open(name).convert('L')
    (w,h) = x.size
    wp = int(256.0*w/min(w,h))
    hp = int(256.0*h/min(w,h))

    x = x.resize((wp,hp),Image.BILINEAR)
    (w,h) = x.size

    if h > w:
        center = h/2
        x = x.crop((0, center - 128,
                    256, center + 128))
    elif h < w:
        center = w/2
        x = x.crop((center - 128, 0,
                    center + 128, 256))
    (w,h) = x.size
    x = np.array(x,np.uint8).reshape((h,w))/255.0
    x[x > 0.4] = 1
    showImage(x)

    if export:
        if isinstance(export,str):
            exportName = export
        else:
            exportName = name[:name.index('.')] + '-processed.png'
        Image.fromarray(x*255).convert('L').save(exportName)
            
    return x

if __name__ == '__main__':
    processDrawing(sys.argv[1], export = True)
