from __future__ import print_function, division

import numpy as np
import cv2
import os, sys

import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Concatenate
from keras import Model
import keras.backend as K
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os, sys

import os
import shutil
import random

print(keras.__version__, tf.__version__)

from models import QATM, MyNormLayer
from utils import compute_score, locate_bbox

def read_gt( file_path ):
    with open( file_path ) as IN:
        x, y, w, h = [ eval(i) for i in IN.readline().strip().split(',')]
    return x, y, w, h

def create_model( featex, alpha = 1. ):
    T = Input( (None, None, 3), name='template_input' )
    I = Input( (None, None, 3), name='image_input' )
    T_feat = featex(T)
    I_feat = featex(I)
    I_feat, T_feat = MyNormLayer( name='norm_layer' )( [I_feat, T_feat] )
    dist = Lambda( lambda x: tf.einsum( "xabc,xdec->xabde", K.l2_normalize(x[0], axis=-1), K.l2_normalize(x[1], axis=-1) ) , name="cosine_dist")([ I_feat, T_feat ])
    conf_map = QATM(alpha, name='qatm')( dist )
    return Model( [T, I], [conf_map], name='QATM_model' )

vgg19 = keras.applications.vgg19.VGG19( include_top = False, weights = 'imagenet' )

# resize conv3_4 to conv1_2
input_ = vgg19.input
conv1_2 = vgg19.get_layer('block1_conv2').output
conv3_4 = vgg19.get_layer('block3_conv4').output
conv3_4 = Lambda( lambda x: tf.image.resize( x[0], size=(tf.shape(x[1])[1], tf.shape(x[1])[2])), name='resized_image' )( [conv3_4, conv1_2] )
concat = Concatenate()( [conv1_2, conv3_4] )
featex = Model( [input_], [concat], name='featex' )
# resize conv1_2 to conv3_4, used when image size is too big
input_ = vgg19.input
conv1_2 = vgg19.get_layer('block1_conv2').output
conv3_4 = vgg19.get_layer('block3_conv4').output
conv1_2 = Lambda( lambda x: tf.image.resize( x[1], size=(tf.shape(x[0])[1], tf.shape(x[0])[2])), name='resized_image' )( [conv3_4, conv1_2] )
concat = Concatenate()( [conv1_2, conv3_4] )
featex2 = Model( [input_], [concat], name='featex2' )
model = create_model( featex , alpha=25)
model_bkup = create_model( featex2 , alpha=25)


def run_files(image_path, image_size, template_path, template_size = 100, outfile='results.png', rect = ( (0,0), (0,0)), title='QATM Piece Match', verbose = True):

    if verbose:
        print("Processing: %s  %s  " %(image_path,template_path))
        
    original = cv2.imread( template_path )
    m = max(original.shape[0],original.shape[1])
    scale = template_size / m
    resized = cv2.resize(original,  (int(original.shape[0]*scale),int(original.shape[1]*scale)) , interpolation = cv2.INTER_NEAREST )
    template = resized[...,::-1]
    w, h = (template.shape[0],template.shape[1])

    image = cv2.imread( image_path )[...,::-1]
    m = max(image.shape[0],image.shape[1])
    scale = image_size / m
    image = cv2.resize(image,  (int(image.shape[1]*scale),int(image.shape[0]*scale)) , interpolation = cv2.INTER_NEAREST )

    # process images
    template_ = np.expand_dims(preprocess_input( template ), axis=0)
    image_ = np.expand_dims(preprocess_input( image ) , axis=0)
    if w*h <= 4000:
        val = model.predict( [template_, image_] )
    else:
        # used when image is too big
        val = model_bkup.predict( [template_, image_] )

    # compute geometry average on score map
    val = np.log( val )
    gray = val[0,:,:,0]
    gray = cv2.resize( gray, (image.shape[1], image.shape[0]) )
    score = compute_score( gray, w, h ) 
    score[score>-1e-7] = score.min()
    score = np.exp(score / (h*w)) # reverse number range back after computing geometry average
    
    # plot result
    x, y, ww, hh = locate_bbox( score, w, h )
    image_plot = cv2.rectangle( image.copy(), (int(x), int(y)), (int(x+ww), int(y+hh)), (255, 0, 0), 3 )
    p1 =  (  ( int((rect[0][0] + rect[1][0])*0.5)) , int((rect[0][1] + rect[1][1])*0.5) )
    p2 =  ( int(x+ww*0.5), int(y+hh*0.5) )
    image_plot = cv2.line( image_plot, p1, p2, (255, 255, 255), 2 )
    image_plot = cv2.rectangle( image_plot, rect[0], rect[1], (0, 0, 255), 3 )
    image_gt = cv2.rectangle( image.copy(), rect[0], rect[1], (0, 0, 255), 3 )
    fig, ax = plt.subplots( 1, 4, figsize=(20, 5) )

    ax[0].imshow(image_gt)
    ax[0].set_title('Ground Truth Blue')


    ax[1].imshow(template)
    ax[1].set_title('Piece')

    ax[2].imshow(image_plot)
    ax[2].set_title('Ground Truth Blue - Best Guess Red')

    ax[3].imshow(score, 'jet')
    ax[3].set_title('Per Pixel Score')

    fig.suptitle(title, fontsize=20)
    plt.savefig(outfile)
    fig.clf()


def singleFile(image, template_dir, basename, results_dir, x, y, w, h, imageSize=600, size=50, sizeX=50, sizeY=50, force = False):
    title="QATM Piece Match (%s : (%i,%i)) " % (basename, x, y)  
    rect = ( (int(x*sizeX), int( (h-y) * sizeY)),   ((int((x+1)*sizeX)), int((h-y-1) * sizeY)) )
    template = os.path.join(template_dir, basename + '_'+str(x)+'_'+str(y)+'.png')
    result = os.path.join(results_dir, os.path.splitext(os.path.basename(template))[0] + '_result' + os.path.splitext(os.path.basename(template))[1])
    if os.path.exists(result) and not force:
        return
    else:
        run_files(image, imageSize, template, size, result, rect, title)

def runPieces(basename, image, imageSize, template_dir, templateSize, results_dir, w, h, force=False, dryRun=False):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    tmp = cv2.imread( image )

    (height, width, _) = tmp.shape
    m = max(height,width)
    scale = imageSize / m    
    sizeX = width * scale / w
    sizeY = height * scale / h

    count = 1
    for x in range(0,w):
        for y in range(0,h):
            singleFile(image, template_dir, basename, results_dir, x, y, w, h, imageSize, templateSize, sizeX,sizeY, force)
            if dryRun:
                break
        if dryRun:
            break


dryrun = False
force = True
imageSize = 600
templateSize = 40


runPieces('princess', '../puzzle/puzzles/princess/12_15/princess_image.jpg', imageSize,  '../puzzle/puzzles/princess/12_15/piece', templateSize, 'puzzles/princess/12_15/pieces_results', 12, 15, force, dryrun)
runPieces('princess_trimmed', '../puzzle/puzzles/princess/12_15/princess_image.jpg', imageSize,  '../puzzle/puzzles/princess/12_15/trimmed', templateSize, 'puzzles/princess/12_15/trimmed_results', 12, 15, force, dryrun)


