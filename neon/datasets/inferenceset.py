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
#        if not os.path.exists(repofile):
#            if leafdir == 'train':
#               self.download_to_repo(self.raw_train_url, rootdir)
#            else:
#               self.download_to_repo(self.raw_test_url, rootdir)
#            infile = zipfile.ZipFile(repofile)
#            infile.extractall(rootdir)
#            infile.close()
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
            #img = transform.resize(img, (self.image_width, self.image_width))

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
    def load(self, backend=None, experiment=None):

        # We only need test set
        if 'repo_path' not in self.__dict__:
            raise AttributeError('repo_path not specified in config')

        self.repo_path = os.path.expandvars(os.path.expanduser(self.repo_path))

        rootdir = self.repo_path

        (self.inputs['train'], imagelist, imgdims) = self.read_images(rootdir, 'train')
        (self.inputs['test'], imagelist, imgdims) = self.read_images(rootdir, 'test')

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

