import numpy

import chainer
from chainer import function_node


class PersistentMemoryFunction(function_node.FunctionNode):

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        x, memory = inputs
        batch = len(x)

        if (isinstance(x, numpy.ndarray) and
                not (x.flags.c_contiguous or x.flags.f_contiguous) and
                1 in x.shape):
            x = numpy.ascontiguousarray(x)

        y = x.dot(memory).astype(x.dtype, copy=False)

        # self.retain_outputs((0,))
        return y

    def backward(self, indexes, grad_outputs):
        x, memory = self.get_retained_inputs()
        gy, = grad_outputs

        ret = []
        gx, = MemoryGradData().apply((memory, gy))
        ret.append(chainer.functions.cast(gx, x.dtype))

        gm, = MemoryGradWeight().apply((x, gy))
        ret.append(chainer.functions.cast(gm, memory.dtype))

        return ret


class MemoryGradData(function_node.FunctionNode):

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        memory, gy = inputs

        if (isinstance(gy, numpy.ndarray) and
                not (gy.flags.c_contiguous or gy.flags.f_contiguous) and
                1 in gy.shape):
            gy = numpy.ascontiguousarray(gy)

        gx = gy.dot(memory).astype(gy.dtype, copy=False)
        return gx,

    def backward(self, indexes, grad_outputs):
        memory, gy = self.get_retained_inputs()
        ggx, = grad_outputs

        ret = []
        gm, = MemoryGradWeight().apply((ggx, gy))
        ret.append(chainer.functions.cast(gm, memory.dtype))
        ggy = persistent_memory(ggx, memory)
        ret.append(chainer.functions.cast(ggy, gy.dtype))
        return ret


class MemoryGradWeight(function_node.FunctionNode):

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs

        if (isinstance(gy, numpy.ndarray) and
                not (gy.flags.c_contiguous or gy.flags.f_contiguous) and
                1 in gy.shape):
            gy = numpy.ascontiguousarray(gy)

        gW = gy.T.dot(x).astype(gy.dtype, copy=False)
        return gW,

    def backward(self, indexes, grad_outputs):
        x, gy = self.get_retained_inputs()
        ggW, = grad_outputs

        ret = []
        gx, = MemoryGradData().apply((ggW, gy))
        ret.append(chainer.functions.cast(gx, x.dtype))
        ggy = persistent_memory(x, ggW)
        ret.append(chainer.functions.cast(ggy, gy.dtype))
        return ret


def persistent_memory(x, memory):
    x = x.reshape(len(x), -1)
    args = x, memory
    y, = PersistentMemoryFunction().apply(args)
    return y
