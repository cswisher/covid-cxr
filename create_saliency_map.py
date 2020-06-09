
#boilerplate imports.
import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import PIL.Image
from matplotlib import pylab as P
import pickle
import os
import glob
import sys
import pandas as pd
# slim=tf.contrib.slim
# 1. Install tf_slim using: pip install git+https://github.com/adrianc-a/tf-slim.git@remove_contrib
# 2. Replace imports of slim with import tf_slim as slim 
#    in the models/research/slim folder - in inception_v3.py and inception_utils.py.
import tf_slim as slim
import matplotlib.pyplot as plt
import pydicom

# From our repository.
import saliency

def load_model():
    ## Setup COVID model
    weightspath = '/home/cswisher/COVID-Net/COVID-Net-Large'   
    metaname    = 'model.meta_eval'
    ckptname    = 'model-2069'
    
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


    return image_tensor, pred_tensor, sess, prediction, xrai_object, neuron_selector 


## Utils
def load_img(imagepath, ftype='dcm'):
    if ftype =='dcm':
        x = pydicom.dcmread(imagepath).pixel_array
        x = cv2.cvtColor(x,cv2.COLOR_GRAY2RGB)
    else:
        x = cv2.imread(imagepath)
    x = cv2.resize(x, (224, 224))
    
    if int(x.max()) < 1.0:
        bit_depth = 1.0
    elif int(x.max()) < 2^8:
        bit_depth = 255.0
    elif int(x.max()) < 2048:
        bit_depth = 2048.0
    else:
        bit_depth = 4096.0

    x = x.astype('float32') / bit_depth

    return x, np.expand_dims(x, axis=0)


def get_xrai(x, xrai_object, neuron_selector, class_no):

    # Compute XRAI attributions with default parameters
    xrai_attributions_covid = xrai_object.GetMask(x, feed_dict={neuron_selector: class_no})

    return xrai_attributions_covid#, xrai_attributions_pneum

def make_heat_map(img, overlay, prob, title, correct_prob = True, alpha = 0.2, colormap = plt.cm.seismic, clims_off = False, interp = True):
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
        if correct_prob:
            plt.imshow(dst*prob,cmap=colormap, alpha=alpha, clim=[min_dst, dst.max()])
        else:
            plt.imshow(dst,cmap=colormap, alpha=alpha, clim=[min_dst, dst.max()])
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

def create_overlay_img(outfname, pred, x, xrai_covid, inv_mapping, correct_prob = True, make_original = True):
    print(outfname)
    if make_original:
        plt.figure(figsize = (16,16), dpi=300)
        plt.imshow(x)
        make_heat_map(x, xrai_covid, pred[0][2], 'COVID-19 Saliency Map', colormap = plt.cm.gist_heat, correct_prob = True)
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

def create_saliency_map(data_path, out_dir, ftype = 'dcm', correct_prob = True):


    fnames = [f for f in glob.iglob(data_path + '**/*.' + ftype, recursive=True)]
   
    mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
    inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}

    image_tensor, pred_tensor, sess, prediction, xrai_object, neuron_selector = load_model()

    for f in fnames:
        print(f)
        # Load Imag
        try:
            x, im = load_img(f, ftype)
        except:
            print('Skipping {}'.format(f))
            continue
        # Inference
        pred = sess.run(pred_tensor, feed_dict={image_tensor: im})
        prediction_class = pred.argmax(axis=1)[0]
        
        xrai_covid = get_xrai(x, xrai_object, neuron_selector, 2)
        xrai_pneum = get_xrai(x, xrai_object, neuron_selector, 1) 
       
        if ftype == 'dcm':
             pid = str(pydicom.dcmread(f).PatientName) + '-'
        else:
             pid = ''

        outfname = os.path.join(out_dir, pid + f.split('/')[-1].split(".dcm")[0]  + '-Probabilities_covid{:.3f}_pneum{:.3f}.png'.format(pred[0][2], pred[0][1]))
        
        create_overlay_img(outfname, pred, x, xrai_covid, inv_mapping, correct_prob = correct_prob)
        create_overlay_img(outfname, pred, x, xrai_pnuem, inv_mapping, correct_prob = correct_prob, make_original = False)
    
    


if __name__ == "__main__":
    print(sys.argv)
    data_path = sys.argv[1]
    out_dir = sys.argv[2]
    correct_prob = bool(sys.argv[3])
    create_saliency_map(data_path, out_dir, correct_prob=correct_prob)
# python create_saliency_map.py /home/cswisher/data/rochester/ /home/cswisher/data/rochester/
