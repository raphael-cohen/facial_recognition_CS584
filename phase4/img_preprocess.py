from os import listdir
import os
from os.path import isfile, join
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# mypath='facescrub-master/download'
mypath = '/media/Download/facescrub-master/download'

folders = [name for name in os.listdir(mypath) if os.path.isdir(mypath+'/'+name)]

for f in folders:

    npath = join(mypath,f,'face')
    onlyfiles = [ p for p in listdir(npath) if isfile(join(npath,p)) ]
    images = np.empty((len(onlyfiles),64,64), dtype=object)
    # images = np.empty(len(onlyfiles), dtype=object)

    to_delete = []
    for n in range(0, len(onlyfiles)):
        try:
            images[n] = cv2.cvtColor(cv2.resize( cv2.imread( join(npath,onlyfiles[n]) ), (64, 64)), cv2.COLOR_BGR2GRAY).astype('float')
            # images[n] = images[n].astype('float')

        except:
            print("\nmarche pas: \n")
            to_delete.append(n)

    # images = images.reshape((len(onlyfiles), 64, 64, 1))

    print(to_delete)
    rows = int(len(onlyfiles) - len(to_delete))
    images = np.delete(images, to_delete, axis = 0)

    images = images / 255.

    images = images.reshape((rows, 64,64,1))
    print(images.shape)
    plt.imshow((images[0].astype(np.float32)).reshape((64,64)))

    plt.show()
    # np.save(join('np_faces',f),images)
    # img = np.load(join('np_faces',f+'.npy'))
    # cv2.imshow('color_image',(images[0]).reshape(64,64))
    # cv2.waitKey(0)                 # Waits forever for user to press any key
    # cv2.destroyAllWindows()

    print("{0}: {1}".format(f, len(onlyfiles)))
