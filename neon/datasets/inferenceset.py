import cPickle
import glob
import logging
import numpy as np
import os
from skimage import io, transform
import zipfile
import copy

from neon.datasets.dataset import Dataset
from neon.util.param import opt_param, req_param
from neon.util.persist import deserialize


logger = logging.getLogger(__name__)

class Inferenceset(Dataset):
    """
    Sets up an dataset for inference
    """
    
    def __init__(self, **kwargs):

        self.__dict__.update(kwargs)

    def read_images(self, rootdir, leafdir, wildcard=''):
        
        logger.info('Reading images from %s', leafdir)
        repofile = os.path.join(rootdir, leafdir)
        
        dirname = os.path.join(rootdir, leafdir, wildcard)

        imagelist = []
        imagecount = 0
        
        for walkresult in os.walk(dirname):
            for filename in walkresult[2]:
                imagelist.append(os.path.join(dirname, filename))
                imagecount += 1
        imagelist.sort()

        self.framesize = self.image_width * self.image_width

        self.imagesize = self.nchannels * self.framesize
        inputs = np.zeros((imagecount, self.imagesize), dtype=np.float32)
        #targets = np.zeros((imagecount, 121), dtype=np.float32)
        imgdims = np.zeros(imagecount)

        for imageind, filename in enumerate(imagelist):
            img = io.imread(filename, as_grey=False)

            # convert RGB to BGR to coordinate with caffe
            r_col = copy.deepcopy(img[:,:,0])
            img[:,:,0] = copy.deepcopy(img[:,:,2])
            img[:,:,2] = copy.deepcopy(r_col)
            imgdims[imageind] = np.mean(img.shape)

            img = np.float32(img)
            
            for c in np.arange(self.nchannels):
                for w in np.arange(self.image_width):
                    for h in np.arange(self.image_width):
                        inputs[imageind][c*self.framesize + w*self.image_width + h] = img[w][h][c]
            #inputs[imageind] = img.ravel()

#        for classind in range(nclasses):
#            for filename in filetree[classind]:
#                img = io.imread(filename, as_grey=True)
#                imgdims[imageind] = np.mean(img.shape)
#                img = transform.resize(img, (self.image_width,
#                                             self.image_width))
#                img = np.float32(img)
#                # Invert the greyscale.
#                img = 1.0 - img
#                inputs[imageind][:self.framesize] = img.ravel()
#                inputs[imageind][self.framesize:] = self.whiten(filename, img).ravel()
#                targets[imageind, classind] = 1
#                imageind += 1
        return inputs, imagelist, imgdims

    def read_flots(self, rootdir, leafdir, wildcard=''):

        logger.info('Reading float number inputs from %s', leafdir)
        repofile = os.path.join(rootdir, leafdir)

        dirname = os.path.join(rootdir, leafdir, wildcard)

        input_list = []
        input_cnt = 0

        # read in every files in this directory and batch them
        for walkresult in os.walk(dirname):
            for filename in walkresult[2]:
                input_list.append(os.path.join(dirname, filename))
                input_cnt += 1

        input_list.sort()

        inputs = np.empty((1, 440), dtype=np.float32) # NOTE: assume every input as of size 440

        # Now parse the input and batch them together
        for filename in input_list:
            
            # The first line gives the dimensions of the input
            with open(filename) as f:
                dim = [int(s) for s in f.readline().split() if s.isdigit()]

            input_size = dim[0] * dim[1] * dim[2] * dim[3]
            input_arr = np.empty((dim[0], dim[1]), dtype=np.float32)

            # read line by line and fill in the array
            with open(filename) as f:
                f.readline() # skip the first line
                for x in np.nditer(arr, op_flags=['readwrite'], order='C'):
                    x[...] = float(f.readline().strip())

            # concantante with the total input
            inputs = np.concatenate(inputs, input_arr, axis=0)
        
        # All inputs concatenate together
        # remove the first row which is a place holder
        inputs = np.delete(inputs, obj=0, axis=0)

        return inputs

    def load(self, backend=None, experiment=None):

        # We only need test set
        if 'repo_path' not in self.__dict__:
            raise AttributeError('repo_path not specified in config')

        self.repo_path = os.path.expandvars(os.path.expanduser(self.repo_path))
        rootdir = self.repo_path

        if 'input_type' not in self.__dict__:
            raise AttributeError('input_type not specified in config (image or floats)')

        if str(self.input_type) == 'image':
            (self.inputs['train'], imagelist, imgdims) = self.read_images(rootdir, 'train')
            (self.inputs['test'], imagelist, imgdims) = self.read_images(rootdir, 'test')
        elif str(self.input_type) == 'floats':
            (self.inputs['test']) = self.read_floats(rootdir, 'test')
        else:
            raise NotImplementedError('input type %s not implemented', str(self.input_type))

    def get_mini_batch(self, batch_idx=0):

        # return batched images
        bs = self.batch_size
        betype = self.backend_type

        # Allocate tensor for the corresponding backend
        self.inp_be = self.backend.empty((self.imagesize, bs))

        # Make the input in numpy array
        # first transpose the input
        # NOTE: Hardcoded float32
        inputs = np.zeros((bs, self.imagesize), dtype=np.float32) 

        # get how many images you have
        image_cnt = self.inputs['test'].shape[0]

        for i in np.arange(bs):
            inputs[i] = copy.deepcopy(self.inputs['test'][i%image_cnt])

        # Matrix populated, copy over to the backend tensor
        self.inp_be.copy_from(inputs.T)

        # populate labels randomly
        # super hacky...
        labels = np.zeros((1000, bs))

        for i in np.arange(bs):
            labels[0][i] = 1

        # convert it to backend tensor
        self.lbl_be = self.backend.empty((1000, bs))
        self.lbl_be.copy_from(labels)

        return self.inp_be, None, {'l_id':self.lbl_be}

    def process_result(self, results):
        
        # Results is a numpy array
        if str(self.input_type) == 'image':
            # get the class with the maximum probability
            image_class = np.argmax(results)
            # print out the result
            logger.info("Image class lable is %s", str(image_class))

