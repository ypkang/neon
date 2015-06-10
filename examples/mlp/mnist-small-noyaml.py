# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
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
Example that creates and uses a network without a configuration file.
"""

import logging, logging.handlers
from neon.backends import gen_backend
from neon.layers import FCLayer, DataLayer, CostLayer
from neon.models import MLP
from neon.transforms import RectLin, Logistic, CrossEntropy
from neon.datasets import MNIST
from neon.experiments import FitPredictErrorExperiment

logging.basicConfig(level=20)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# if we set handler to DEBUG but don't set logger to debug
# handler will still not use debug

# declare a socketHandler
socketHandler = logging.handlers.SocketHandler('localhost',
                    logging.handlers.DEFAULT_TCP_LOGGING_PORT)


#create file handler which logs all messages
fh = logging.FileHandler('spam.log')
fh.setLevel(logging.DEBUG)

logger.addHandler(fh)
logger.addHandler(socketHandler)

logging.info("we've initialized the loggers")

def create_model(nin):
    layers = []
    layers.append(DataLayer(nout=nin))
    layers.append(FCLayer(nout=100, activation=RectLin()))
    layers.append(FCLayer(nout=10, activation=Logistic()))
    layers.append(CostLayer(cost=CrossEntropy()))
    model = MLP(num_epochs=10, batch_size=128, layers=layers)
    return model


def run():
    model = create_model(nin=784)
    backend = gen_backend(rng_seed=0)
    dataset = MNIST(repo_path='~/data/')
    experiment = FitPredictErrorExperiment(model=model,
                                           backend=backend,
                                           dataset=dataset)
    experiment.run()


if __name__ == '__main__':
    run()
