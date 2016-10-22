from __future__ import division
import numpy as np
from PIL import Image
from StringIO import StringIO
import urllib2
from urlparse import urlparse
import os

# Functions to process RGB images and make them become feature vectors

def process_image_file(image_path):
    '''Given an image path it returns its feature vector.

    Args:
      image_path (str): path of the image file to process.

    Returns:
      list of float: feature vector on success, None otherwise.
    '''
    image_fp = StringIO(open(image_path, 'rb').read())
    try:
        image = Image.open(image_fp)
        return process_image(image)
    except IOError:
        return None


def process_image(image, blocks=6):
    '''Given a PIL Image object it returns its feature vector.

    Args:
      image (PIL.Image): image to process.
      blocks (int, optional): number of block to subdivide the RGB space into.

    Returns:
      list of float: feature vector if successful. None if the image is not
      RGB.
    '''
    if not image.mode == 'RGB':
        return None
    feature = [0] * blocks * blocks * blocks
    pixel_count = 0
    for pixel in image.getdata():
        ridx = int(pixel[0]/(256/blocks))
        gidx = int(pixel[1]/(256/blocks))
        bidx = int(pixel[2]/(256/blocks))
        idx = ridx + gidx * blocks + bidx * blocks * blocks
        feature[idx] += 1
        pixel_count += 1
    return [x/pixel_count for x in feature]


# In[ ]:

# Predict classes for all images in the data folder

picturesFolder = './data/train_photos'
desinationFolder = './data/featurePictures'

originals = [f for f in os.listdir(picturesFolder) if os.path.isfile(os.path.join(picturesFolder, f))]
processed = [f for f in os.listdir(desinationFolder) if os.path.isfile(os.path.join(desinationFolder, f))]

def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]
    
processables = [s.replace('.npy', '') for s in diff([i + '.npy' for i in originals], processed)]

print 'Missing', len(processables)

processed = 1

for pic in processables:
	featureArray = np.array(process_image_file(os.path.join(picturesFolder, pic)))
    	np.save(os.path.join(desinationFolder, pic), featureArray)
	print 'Processed', processed
	processed += 1
