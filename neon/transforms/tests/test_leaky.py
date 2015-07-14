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
from nose.plugins.attrib import attr
import numpy as np

from neon.backends.cpu import CPU, CPUTensor
from neon.transforms.rectified import RectLeaky
from neon.util.testing import assert_tensor_equal


def compare_cpu_tensors(inputs, outputs, deriv=False):
    rlin = RectLeaky()
    be = CPU()
    temp = be.zeros(inputs.shape)
    if deriv is True:
        rlin.apply_derivative(be, CPUTensor(inputs), temp)
    else:
        rlin.apply_function(be, CPUTensor(inputs), temp)
    be.subtract(temp, CPUTensor(outputs), temp)
    assert_tensor_equal(temp, be.zeros(inputs.shape))


def compare_cc2_tensors(inputs, outputs, deriv=False):
    from neon.backends.cc2 import GPU, GPUTensor
    rlin = RectLeaky()
    be = GPU()
    temp = be.zeros(inputs.shape)
    if deriv is True:
        rlin.apply_derivative(be, GPUTensor(inputs), temp)
    else:
        rlin.apply_function(be, GPUTensor(inputs), temp)
    be.subtract(temp, GPUTensor(outputs), temp)
    assert_tensor_equal(temp, be.zeros(inputs.shape))


def test_rectleaky_positives():
    inputs = np.array([1, 3, 2])
    outputs = np.array([1, 3, 2])
    compare_cpu_tensors(inputs, outputs)


def test_rectleaky_negatives():
    inputs = np.array([[-1, -3], [-2, -4]])
    outputs = np.array([[-0.01, -0.03], [-0.02, -0.04]])
    compare_cpu_tensors(inputs, outputs)


def test_rectleaky_mixed():
    inputs = np.array([[4, 0], [-2, 9]])
    outputs = np.array([[4, 0], [-0.02, 9]])
    compare_cpu_tensors(inputs, outputs)


@attr('cuda')
def test_rectleaky_cc2tensor():
    inputs = np.array([[4, 0], [-2, 9]])
    outputs = np.array([[4, 0], [-0.02, 9]])
    compare_cc2_tensors(inputs, outputs)


def test_rectleaky_derivative_positives():
    inputs = np.array([1, 3, 2])
    outputs = np.array([1, 1, 1])
    compare_cpu_tensors(inputs, outputs, deriv=True)


def test_rectleaky_derivative_negatives():
    inputs = np.array([[-1, -3], [-2, -4]])
    outputs = np.array([[0.01, 0.01], [0.01, 0.01]])
    compare_cpu_tensors(inputs, outputs, deriv=True)


def test_rectleaky_derivative_mixed():
    inputs = np.array([[4, 0], [-2, 9]])
    outputs = np.array([[1, 0.01], [0.01, 1]])
    compare_cpu_tensors(inputs, outputs, deriv=True)


@attr('cuda')
def test_rectleaky_derivative_cc2tensor():
    inputs = np.array([[4, 0], [-2, 9]])
    outputs = np.array([[1, 0.01], [0.01, 1]])
    compare_cc2_tensors(inputs, outputs, deriv=True)


def test_rectleaky_slope_zero_rectlin_equiv():
    be = CPU()
    inputs = be.uniform(low=-5.0, high=10.0, size=(10, 10))
    lin_buf = be.empty(inputs.shape)
    leaky_buf = be.empty(inputs.shape)
    be.rectlin(inputs, out=lin_buf)
    be.rectleaky(inputs, slope=0.0, out=leaky_buf)
    assert_tensor_equal(lin_buf, leaky_buf)


def test_rectleaky_derivative_slope_zero_rectlin_equiv():
    be = CPU()
    inputs = be.uniform(low=-5.0, high=10.0, size=(10, 10))
    lin_buf = be.empty(inputs.shape)
    leaky_buf = be.empty(inputs.shape)
    be.rectlin_derivative(inputs, out=lin_buf)
    be.rectleaky_derivative(inputs, slope=0.0, out=leaky_buf)
    assert_tensor_equal(lin_buf, leaky_buf)


@attr('cuda')
def test_cc2_rectleaky_slope_zero_rectlin_equiv():
    from neon.backends.cc2 import GPU
    be = GPU()
    inputs = be.uniform(low=-5.0, high=10.0, size=(10, 10))
    lin_buf = be.empty(inputs.shape)
    leaky_buf = be.empty(inputs.shape)
    be.rectlin(inputs, out=lin_buf)
    be.rectleaky(inputs, slope=0.0, out=leaky_buf)
    assert_tensor_equal(lin_buf, leaky_buf)


@attr('cuda')
def test_cc2_rectleaky_derivative_slope_zero_rectlin_equiv():
    from neon.backends.cc2 import GPU
    be = GPU()
    inputs = be.uniform(low=-5.0, high=10.0, size=(10, 10))
    lin_buf = be.empty(inputs.shape)
    leaky_buf = be.empty(inputs.shape)
    be.rectlin_derivative(inputs, out=lin_buf)
    be.rectleaky_derivative(inputs, slope=0.0, out=leaky_buf)
    assert_tensor_equal(lin_buf, leaky_buf)
