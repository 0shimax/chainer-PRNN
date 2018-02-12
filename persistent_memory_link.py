import functools
import operator

from chainer import initializers
from chainer import link
from chainer import variable
import chainer.functions as F

from persistent_memory_function import persistent_memory


class PersistentMemory(link.Chain):

    def __init__(self, in_size, slot_size, memory_size, initialW=None):
        """
        in_size: hidden_state h_size
        """
        super().__init__()

        self.slot_size = slot_size
        self.memory_size = memory_size

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.memory = variable.Parameter(W_initializer)
            self.projection_matrix = variable.Parameter(W_initializer)

    def _initialize_params(self, in_size):
        self.memory.initialize((self.memory_size, self.slot_size))
        self.projection_matrix.initialize((in_size, self.memory_size))

    def calculate_memory_weight(self, in_size, hidden_state):
        # print("hidden_state", hidden_state)
        DM = F.matmul(self.projection_matrix, self.memory)  # (in_size, slot_size)
        # print("DM----", DM)
        sim = F.matmul(hidden_state, DM)  # (batch_size, slot_size)
        # print("sim----", sim)
        n_batch, n_slot = sim.shape
        normed_hidden = F.reshape(F.batch_l2_norm_squared(hidden_state), (-1, 1))
        sim = F.exp(F.log(sim) - F.log(1+F.tile(normed_hidden, (1, n_slot))))
        # sim /= F.tile(normed_hidden, (1, n_slot))  # (batch_size, slot_size)/(batch_size,)
        sim = F.exp(F.log(sim) - F.log(1+F.tile(F.sum(DM*DM, axis=0), (n_batch, 1))))
        # sim /= F.tile(
        #     F.sum(DM*DM, axis=0), (n_batch, 1))  # (batch_size, slot_size)/(slot_size,)
        return F.softmax(sim)  # (batch_size, slot_size)

    def __call__(self, x):
        in_size = None
        if self.memory.data is None:
            in_size = functools.reduce(operator.mul, x.shape[1:], 1)
            self._initialize_params(in_size)
        self.weight = self.calculate_memory_weight(in_size, x)

        n_batch, n_slot = self.weight.shape
        n_memory, _ = self.memory.shape
        # (batch_size, slot_size)*(memory_size, slot_size)
        wm = F.reshape(
                F.tile(self.weight, (1, n_memory)), (-1, n_memory, n_slot)) \
            * F.tile(self.memory, (n_batch, 1, 1))
        return F.sum(wm, axis=2)
