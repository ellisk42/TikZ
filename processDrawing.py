import numpy as np
from PIL import Image
from utilities import showImage

def processDrawing(name, export = False):
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
    x[x > 0.3] = 1
    showImage(x)

    if export:
        if isinstance(export,str):
            exportName = export
        else:
            exportName = name[:name.index('.')] + '-processed.png'
        Image.fromarray(x*255).convert('L').save(exportName)
            
    return x
processDrawing("drawings/hand1.jpg", export = True)
processDrawing("drawings/hand2.jpg", export = True)
processDrawing("drawings/hand3.jpg", export = True)
