import numpy as np
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import measure
from skimage.segmentation import random_walker
import matplotlib.pyplot as plt
from scipy import ndimage

# Generate an initial image with two overlapping circles
x, y = np.indices((80, 80))
x1, y1, x2, y2 = 28, 28, 44, 52
r1, r2 = 16, 20
mask_circle1 = (x - x1) ** 2 + (y - y1) ** 2 < r1 ** 2
mask_circle2 = (x - x2) ** 2 + (y - y2) ** 2 < r2 ** 2
image = np.logical_or(mask_circle1, mask_circle2)




def processimage(imgsource):

    #ici modele de binarisation de l'image
    #zbinary = np.zeros(imagrey.shape) #image de zeros de meme taille que image
    #image=np.ma.masked_array(zbinary, mask=imagrey<40,fill_value=1)

    jpgimagey8 = imgsource[200:600,156:872] #fenetrage
    #on voit l'image binaire
    plt.imshow(jpgimagey8, cmap='gray', interpolation='nearest')
    plt.show()

    imageseuil = jpgimagey8<180
    plt.imshow(imageseuil, cmap='gray', interpolation='nearest')
    plt.show()

    # Now we want to separate the two objects in image
    #    Generate the markers as local maxima of the distance
    # to the background
    distance = ndimage.distance_transform_edt(imageseuil)

    plt.imshow(distance, cmap='gray', interpolation='nearest')
    plt.show()


    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=imageseuil)
    markers = measure.label(local_maxi)
    labels_ws = watershed(-distance, markers, mask=imageseuil)

    plt.imshow(labels_ws, cmap='spectral', interpolation='nearest')
    plt.show()

    markers[~image] = -1
    labels_rw = random_walker(imageseuil, markers)

    plt.figure(figsize=(12, 3.5))
    plt.subplot(141)
    plt.imshow(jpgimagey8, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.title('image')
    plt.subplot(142)
    plt.imshow(-distance, interpolation='nearest')
    plt.axis('off')
    plt.title('distance map')
    plt.subplot(143)
    plt.imshow(labels_ws, cmap='spectral', interpolation='nearest')
    plt.axis('off')
    plt.title('watershed segmentation')
    plt.subplot(144)
    plt.imshow(labels_rw, cmap='spectral', interpolation='nearest')
    plt.axis('off')
    plt.title('random walker segmentation')

    plt.tight_layout()
    plt.show()



#read avi with ffmpeg
import subprocess as sp


#read avi from pipe

import numpy
import sys #pour prc

def getfileparameters(filename):
    command = [FFMPEG_BIN,  '-i', filename, '-f image2', '-']
    pipe = sp.Popen(command, stdout=sp.PIPE , stderr = sp.PIPE)
    stdout_iterator = iter(pipe.stderr.readline, b"")
    for laline in stdout_iterator:  #sort dans stderr
        print(laline)
    pipe.terminate()





#le main
FFMPEG_BIN = r"c:\tmp\ffmpeg.exe"
videofile=r"F:\videos\multifish\30fish-MirrorTank-HD.avi"

getfileparameters(videofile)

command = [ FFMPEG_BIN,
            '-i', videofile,
            '-f','image2pipe',
            '-r','1',
            '-pix_fmt','rgb24',
           '-vcodec','rawvideo',
            '-']
pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)

#return_code = pipe.wait()

xsize = 1024
ysize = 768
# read 420*360*3 bytes (= 1 frame)
raw_image = pipe.stdout.read(xsize*ysize*3)
# transform the byte read into a numpy array
image =  numpy.fromstring(raw_image, dtype='uint8')
image = image.reshape((ysize,xsize,3))

processimage(image[:,:,0])

# throw away the data in the pipe's buffer.
pipe.stdout.flush()
