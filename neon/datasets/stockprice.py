# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
mobydick is a character-based text dataset.
The text is available at: http://www.gutenberg.org/ebooks/2701
"""

import numpy as np
import logging
import numpy
import os

from neon.datasets.dataset import Dataset
from neon.util.compat import range
from neon.util.param import opt_param, ensure_dtype

logger = logging.getLogger(__name__)


def normalize(X):
    min = np.min(X, axis=0)
    max = np.max(X, axis=0)
    return (X - min) / (max - min + 1e-6)


class STOCKPRICE(Dataset):

    """
    Sets up AAPL stock price dataset.

    Attributes:
        raw_base_url (str): where to find the source data

        backend (neon.backends.Backend): backend used for this data
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data

    Keyword Args:
        repo_path (str, optional): where to locally host this dataset on disk
    """
    raw_base_url = 'https://www.quandl.com/api/v1/datasets/WIKI/MSFT.csv'

    def __init__(self, **kwargs):
        self.macro_batched = False
        self.__dict__.update(kwargs)

        opt_param(self, ['backend_type'], 'np.float32')
        self.backend_type = ensure_dtype(self.backend_type)  # string to dtype
        logger.info("Setting dtype to" + str(self.backend_type))

    def initialize(self):
        # perform additional setup that can't be done at initial construction
        pass

    def read_txt_file(self, fname):
        """
        Reads the csv file,

        Date, Open, High, Low, Close, Volume, Ex-Dividend, Split Ratio,
        Adj. Open, Adj. High, Adj. Low, Adj. Close, Adj. Volume
        """
        ### TODO: Import csv reader, parse the data
        X = numpy.genfromtxt(fname, delimiter=',')
        X = X[1:, 1:] # ignore dates and labels

        # normalize to 0 through 1
        # TODO: move split parameters to YAML
        X[:6400] = normalize(X[:6400])
        X[6400:7200] = normalize(X[6400:7200])
        return X.T # shape = (n_features, n_examples)

    def transpose_batches(self, data, dtype):
        """
        Transpose each minibatch within the dataset.
        """
        bs = self.data_dim * self.unrolls
        dd = self.data_dim

        if data.shape[0] % bs != 0:
            logger.warning('Incompatible batch size. '
                           'Discarding %d samples...',
                           data.shape[0] % bs)
        nbatches = data.shape[0] / bs
        batchwise = [[] for k in range(nbatches)]
        for batch in range(nbatches):
            batchdata = [self.backend.array(data[(batch * bs + k * dd):
                                                 (batch * bs + (k + 1) *
                                                  dd)], dtype)
                         for k in range(self.unrolls)]
            batchwise[batch] = batchdata
        return batchwise

    def load(self, backend=None, experiment=None):
        self.initialize()
        if self.inputs['train'] is not None:
            return
        if 'repo_path' in self.__dict__:
            self.repo_path = os.path.expandvars(os.path.expanduser(
                                                self.repo_path))
            save_dir = os.path.join(self.repo_path,
                                    self.__class__.__name__)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            train_idcs = list(range(6400))  # 8700 records
            test_idcs = list(range(6400, 7200))
            if 'sample_pct' in self.__dict__:
                if self.sample_pct >= 1.0:
                    self.sample_pct /= 100.0
                    logger.info('sampling pct: %0.2f' % self.sample_pct)
                if self.sample_pct < 1.0:
                    # numpy.random.shuffle(train_idcs)
                    pass
                train_idcs = train_idcs[0:int(8000 * self.sample_pct)]
            url = self.raw_base_url
            name = os.path.basename(url)
            repo_file = os.path.join(save_dir, name)
            if not os.path.exists(repo_file):
                self.download_to_repo(url, save_dir)
            logger.info('loading: %s' % name)
            indat = self.read_txt_file(repo_file)

            self.preinputs = dict()
            self.preinputs['train'] = indat[:, train_idcs]
            self.preinputs['test'] = indat[:, test_idcs]

            for dataset in ('train', 'test'):
                num_batches = self.preinputs[dataset].shape[1]/self.batch_size
                idx_list = numpy.arange(num_batches * self.batch_size)
                idx_list = idx_list.reshape(self.batch_size, num_batches)
                splay_3d = self.preinputs[dataset][:, idx_list.T]
                splay_3d = numpy.transpose(splay_3d, (1, 0, 2))
                splay_3d = splay_3d.reshape(-1, self.batch_size)
                self.inputs[dataset] = splay_3d
                offbyone = numpy.zeros(splay_3d.shape)
                length = offbyone.shape[0]
                offbyone[0:length - self.data_dim, :] = splay_3d[self.data_dim:
                                                                 length, :]
                self.targets[dataset] = offbyone
            self.format(dtype=self.backend_type)  # runs transpose_batches
        else:
            raise AttributeError('repo_path not specified in config')
            # TODO: try and download and read in directly?