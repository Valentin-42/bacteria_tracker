#!/usr/bin/env python

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

def conv(x):
    try:
        return int(x)
    except:
        try:
            return float(x)
        except:
            return x


class Bacteria:
    def __init__(self,h,l):
        self.attributes=dict(zip(h,[conv(v) for v in l]))

    def __getattr__(self,a):
        return self.attributes[a]



data=[l.strip().split(",") for l in open(sys.argv[1],"r").readlines()]
header=data[0]
header=[header[0][1:]]+header[1:]
headermap=dict([(n,i) for i,n in enumerate(header)])
datab=[Bacteria(header,l) for l in data[1:]]
numframes=max([b.frameid for b in datab])+1
numbacteria=max([b.id for b in datab])+1
#datanum=np.array([[conv(v) for i,v in enumerate(l) if i!=1] for l in data[1:]])

summary=np.zeros((numbacteria,numframes),dtype=np.uint8)
orientation=np.ones((numbacteria,numframes),dtype=np.uint8)*255
ellipticity=-np.ones((numbacteria,numframes),dtype=np.float)
for b in datab:
    summary[b.id,b.frameid]=b.state
    orientation[b.id,b.frameid]=b.orientation
    ellipticity[b.id,b.frameid]=np.pi*b.majorL*b.minorL/b.area
dirname=os.path.dirname(sys.argv[1])
basename=os.path.basename(sys.argv[1])
basebase,ext=os.path.splitext(basename)

fig = plt.figure(frameon = True)
fig.set_size_inches(24, 12)
plt.subplot(1,3,1)
plt.imshow(summary)
plt.title("State")
plt.xlabel("Frame")
plt.ylabel("Bacteria")
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(orientation)
plt.title("Orientation")
plt.xlabel("Frame")
plt.ylabel("Bacteria")
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(ellipticity)
plt.title("Ellipticity")
plt.xlabel("Frame")
plt.ylabel("Bacteria")
plt.colorbar()
plt.savefig(os.path.join(dirname,"plot_"+basebase+".png"),facecolor='white',transparent=False,dpi=100)

np.savetxt(os.path.join(dirname,"summary_"+basebase+".mat"),summary,fmt="%d")
np.savetxt(os.path.join(dirname,"orientation_"+basebase+".mat"),orientation,fmt="%d")
np.savetxt(os.path.join(dirname,"ellipticity_"+basebase+".mat"),ellipticity,fmt="%.3f")
