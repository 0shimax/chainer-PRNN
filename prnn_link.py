import functools
import operator

import numpy

from chainer.backends import cuda
from chainer.functions.activation import lstm
from chainer.functions.array import concat
from chainer.functions.array import split_axis
from chainer import initializers
from chainer import link
from chainer.links.connection import linear
from chainer import variable

from persistent_memory_link import PersistentMemory


class LSTMBase(link.Chain):

    def __init__(self, in_size, out_size=None, lateral_init=None,
                 upward_init=None, bias_init=None, forget_bias_init=None,
                 memory_init=None, memory_linear_init=None,
                 slot_size=None, memory_size=None):
        if out_size is None:
            out_size, in_size = in_size, None

        super(LSTMBase, self).__init__()
        if bias_init is None:
            bias_init = 0
        if forget_bias_init is None:
            forget_bias_init = 1
        if memory_init is None:
            memory_init = 1
        self.state_size = out_size
        self.lateral_init = lateral_init
        self.upward_init = upward_init
        self.bias_init = bias_init
        self.forget_bias_init = forget_bias_init
        self.memory_init = memory_init
        self.memory_linear_init = memory_linear_init

        with self.init_scope():
            self.upward = linear.Linear(in_size, 4 * out_size, initialW=0)
            self.lateral = linear.Linear(out_size, 4 * out_size, initialW=0,
                                         nobias=True)
            self.memory_accessing = PersistentMemory(
                in_size, slot_size, memory_size, initialW=0)
            self.memory_linear = linear.Linear(
                memory_size, 4 * out_size, initialW=0)
            if in_size is not None:
                self._initialize_params()

    def _initialize_params(self):
        lateral_init = initializers._get_initializer(self.lateral_init)
        upward_init = initializers._get_initializer(self.upward_init)
        bias_init = initializers._get_initializer(self.bias_init)
        forget_bias_init = initializers._get_initializer(self.forget_bias_init)
        # memory_init = initializers._get_initializer(self.memory_init)
        memory_linear_init = initializers._get_initializer(
            self.memory_linear_init)

        for i in range(0, 4 * self.state_size, self.state_size):
            lateral_init(self.lateral.W.data[i:i + self.state_size, :])
            upward_init(self.upward.W.data[i:i + self.state_size, :])
            memory_linear_init(
                self.memory_linear.W.data[i:i + self.state_size, :])

        a, i, f, o = lstm._extract_gates(
            self.upward.b.data.reshape(1, 4 * self.state_size, 1))

        bias_init(a)
        bias_init(i)
        forget_bias_init(f)
        bias_init(o)


class PRNN(LSTMBase):

    def __init__(self, in_size, out_size=None, lateral_init=None,
                 upward_init=None, bias_init=None, forget_bias_init=None,
                 memory_init=None, memory_linear_init=None,
                 slot_size=None, memory_size=None):
        if out_size is None:
            in_size, out_size = None, in_size
        super().__init__(
            in_size, out_size, lateral_init, upward_init, bias_init,
            forget_bias_init, memory_init, memory_linear_init,
            slot_size, memory_size)
        self.reset_state()

    def to_cpu(self):
        super().to_cpu()
        if self.c is not None:
            self.c.to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super().to_gpu(device)
        if self.c is not None:
            self.c.to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, c, h):
        """Sets the internal state.

        It sets the :attr:`c` and :attr:`h` attributes.

        Args:
            c (~chainer.Variable): A new cell states of LSTM units.
            h (~chainer.Variable): A new output at the previous time step.

        """
        assert isinstance(c, variable.Variable)
        assert isinstance(h, variable.Variable)
        c_ = c
        h_ = h
        if self.xp == numpy:
            c_.to_cpu()
            h_.to_cpu()
        else:
            c_.to_gpu(self._device_id)
            h_.to_gpu(self._device_id)
        self.c = c_
        self.h = h_

    def set_hidden_state(self, h):
        assert isinstance(h, variable.Variable)
        h_ = h
        if self.xp == numpy:
            h_.to_cpu()
        else:
            h_.to_gpu(self._device_id)
        self.h = h_

    def reset_state(self):
        """Resets the internal state.

        It sets ``None`` to the :attr:`c` and :attr:`h` attributes.

        """
        self.c = self.h = None

    def __call__(self, x):
        """Updates the internal state and returns the LSTM outputs.

        Args:
            x (~chainer.Variable): A new batch from the input sequence.

        Returns:
            ~chainer.Variable: Outputs of updated LSTM units.

        """
        if self.upward.W.data is None:
            with cuda.get_device_from_id(self._device_id):
                in_size = functools.reduce(operator.mul, x.shape[1:], 1)
                self.upward._initialize_params(in_size)
                self._initialize_params()

        batch = x.shape[0]
        lstm_in = self.upward(x)
        h_rest = None
        if self.h is not None:
            h_size = self.h.shape[0]
            if batch == 0:
                h_rest = self.h
            elif h_size < batch:
                msg = ('The batch size of x must be equal to or less than'
                       'the size of the previous state h.')
                raise TypeError(msg)
            elif h_size > batch:
                h_update, h_rest = split_axis.split_axis(
                    self.h, [batch], axis=0)
                lstm_in += self.lateral(h_update)
                accessed_memory = self.memory_accessing(h_update)
                lstm_in += self.memory_linear(accessed_memory)
            else:
                lstm_in += self.lateral(self.h)
                accessed_memory = self.memory_accessing(self.h)
                lstm_in += self.memory_linear(accessed_memory)
        if self.c is None:
            xp = self.xp
            with cuda.get_device_from_id(self._device_id):
                self.c = variable.Variable(
                    xp.zeros((batch, self.state_size), dtype=x.dtype))
        self.c, y = lstm.lstm(self.c, lstm_in)

        if h_rest is None:
            self.h = y
        elif len(y.data) == 0:
            self.h = h_rest
        else:
            self.h = concat.concat([y, h_rest], axis=0)

        return y
