
#boilerplate imports.
import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import PIL.Image
from matplotlib import pylab as P
import pickle
import os
import pandas as pd
# slim=tf.contrib.slim
# 1. Install tf_slim using: pip install git+https://github.com/adrianc-a/tf-slim.git@remove_contrib
# 2. Replace imports of slim with import tf_slim as slim 
#    in the models/research/slim folder - in inception_v3.py and inception_utils.py.
import tf_slim as slim
import matplotlib.pyplot as plt


# From our repository.
import saliency


## Setup COVID model
weightspath = '/home/cswisher/COVID-Net/COVID-Net-Large'   
metaname    = 'model.meta_eval'
ckptname    = 'model-2069'

mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}

sess = tf.Session()
tf.get_default_graph()
saver = tf.train.import_meta_graph(os.path.join(weightspath, metaname))
saver.restore(sess, os.path.join(weightspath, ckptname))

graph = tf.get_default_graph()

# Specify inputs and outputs
image_tensor = graph.get_tensor_by_name("input_1:0")
pred_tensor = graph.get_tensor_by_name("dense_3/Softmax:0")

# Construct the scalar neuron tensor.
logits = graph.get_tensor_by_name('dense_3/Softmax:0')
neuron_selector = tf.placeholder(tf.int32)
y = logits[0][neuron_selector]

# Construct tensor for predictions.
prediction = tf.argmax(logits, 1)

## Setup Saliency Objects
# Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
xrai_object = saliency.XRAI(graph, sess, y, image_tensor)

## Load list of files
df = pd.read_csv('/home/cswisher/COVID-Net/test_split_v2.txt', sep = ' ', header=0, names=['id', 'fname','finding'])
print(df.finding.value_counts())
df.head()

## Utils
def load_img(imagepath):
    x = cv2.imread(imagepath)
    x = cv2.resize(x, (224, 224))
    x = x.astype('float32') / 255.0
    return x, np.expand_dims(x, axis=0)


def get_xrai(x):
    # Compute XRAI attributions with default parameters
    xrai_attributions_covid = xrai_object.GetMask(x, feed_dict={neuron_selector: 2})
#    xrai_attributions_pneum = xrai_object.GetMask(x, feed_dict={neuron_selector: 1})
    return xrai_attributions_covid#, xrai_attributions_pneum

def make_heat_map(img, overlay, prob, title, alpha = 0.2, colormap = plt.cm.seismic, clims_off = False, interp = True):
    if interp:
        kernel = np.ones((20,20),np.float32)/25
        dst = cv2.filter2D(overlay,-1,kernel)
    else:
        dst = overlay
    
    plt.imshow(img)
    if clims_off:
        plt.imshow(dst*prob,cmap=colormap, alpha=alpha)
    else:
        if np.percentile(dst,20) == 0: 
            min_dst = 100.0
        else: 
            min_dst = np.percentile(dst,20)
        plt.imshow(dst*prob,cmap=colormap, alpha=alpha, clim=[min_dst, dst.max()])
    
    plt.text(90,235, 'Probability: {:.3f}'.format(prob), bbox=dict(facecolor='grey', alpha=.5))
    plt.title(title)
    plt.axis('off')
    
def make_figure(outfname, pred, x, xrai_covid, xrai_pneum):
    print(outfname)
    
    plt.figure(figsize = (16,12), dpi=200)
    plt.subplot(231)
    plt.imshow(x)
    plt.axis('off')
    plt.text(90,235, 'Prediction: {}'.format(inv_mapping[pred.argmax(axis=1)[0]]),  
             bbox=dict(facecolor='red', alpha=0.5))
    plt.title('Pneumonia Patient')

    plt.subplot(232)
    make_heat_map(x, xrai_covid, pred[0][2], 'COVID-19 Saliency Map', colormap = plt.cm.gist_heat)


    plt.subplot(233)
    make_heat_map(x, xrai_pneum, pred[0][1], 'Non-COVID-19 Pneumonia Saliency Map', colormap = plt.cm.gist_heat)
    plt.tight_layout()

    plt.savefig(outfname, dpi=200,bbox_inches='tight')

    plt.close()

def create_overlay_img(outfname, pred, x, xrai_covid):
    print(outfname)
    
    plt.figure(figsize = (16,16), dpi=300)
    plt.imshow(x)
    make_heat_map(x, xrai_covid, pred[0][2], 'COVID-19 Saliency Map', colormap = plt.cm.gist_heat)
    plt.tight_layout()
    plt.savefig(outfname, dpi=300,bbox_inches='tight')
    plt.close()
        
    plt.figure(figsize = (16,16), dpi=300)
    plt.imshow(x)
    plt.text(90,235, 'Prediction: {}'.format(inv_mapping[pred.argmax(axis=1)[0]]),  
             bbox=dict(facecolor='red', alpha=0.5))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outfname[:-9] + '-Original.png', dpi=300,bbox_inches='tight')
    plt.close()

base_dir    = '/home/cswisher/COVID-Net/data/test/'
img_fname   = df[df.finding == 'COVID-19'].fname.values[1]
imagepath   = os.path.join(base_dir, img_fname) 

for i in range(len(df)):
    if i < 30: continue    
    # Load Image
    imagepath   = os.path.join(base_dir, df.fname[i])
    x, im = load_img(imagepath)
    
    
    # Inference
    pred = sess.run(pred_tensor, feed_dict={image_tensor: im})
    prediction_class = pred.argmax(axis=1)[0]
    outfname = df.finding[i][:5] + '-probs-n{:.3f}_p{:.3f}_c{:.3f}.png'.format(pred[0][0], pred[0][1], pred[0][2])
    
    if (inv_mapping[pred.argmax(axis=1)[0]] != df.finding[i]): continue
                   
    xrai_covid = get_xrai(x)
    
    outfname = 'Finding_'+ df.finding[i][:5] + '-Ptid_' + str(i) + '-Probabilities_covid{:.3f}_pneum{:.3f}.png'.format(pred[0][2], pred[0][1])
    
    create_overlay_img(outfname, pred, x, xrai_covid)




# base_dir    = '/home/cswisher/COVID-Net/data/test/'
# img_fname   = df[df.finding == 'COVID-19'].fname.values[1]
# imagepath   = os.path.join(base_dir, img_fname) 
# 
# start_maps = False
# start_fname = 'ptid040a0743-f663-4746-8224-f0e3bacc7ba5pneum-probs-n0.340_p0.656_c0.004.png'
# for i in range(len(df)):
#     # Load Image
#     imagepath   = os.path.join(base_dir, df.fname[i])
#     x, im = load_img(imagepath)
#     
#     
#     # Inference
#     pred = sess.run(pred_tensor, feed_dict={image_tensor: im})
#     prediction_class = pred.argmax(axis=1)[0]
#     outfname = df.finding[i][:5] + '-probs-n{:.3f}_p{:.3f}_c{:.3f}.png'.format(pred[0][0], pred[0][1], pred[0][2])
#     if outfname != start_fname:
#          start_maps = True
# 
#     if start_maps == False: continue
#     xrai_covid, xrai_pneum = get_xrai(x)
#     
#     outfname = 'ptid' + str(df.id[i]) + df.finding[i][:5] + '-probs-n{:.3f}_p{:.3f}_c{:.3f}.png'.format(pred[0][0], pred[0][1], pred[0][2])
#     
#     make_figure(outfname, pred, x, xrai_covid, xrai_pneum)
# 
