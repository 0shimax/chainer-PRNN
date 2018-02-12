import chainer
import chainer.functions as F
import chainer.links as L

from prnn_link import PRNN


class EncDec(chainer.Chain):

    def __init__(self, in_size, h_unit_size, slot_size, memory_size):
        super().__init__()
        with self.init_scope():
            self.encorder = PRNN(in_size, out_size=h_unit_size,
                                 slot_size=slot_size, memory_size=memory_size)
            self.decorder = PRNN(in_size, out_size=h_unit_size,
                                 slot_size=slot_size, memory_size=memory_size)
            # self.input_to_hidden = L.Linear(dim_obs, h_unit_size)
            self.hidden_to_output = L.Linear(h_unit_size, in_size)
        self.is_train = True

    def reset_state(self):
        self.encorder.reset_state()
        self.decorder.reset_state()

    def set_decorder_init_state(self):
        self.decorder.set_hidden_state(self.encorder.h)

    def train(self, xs, ys):
        for time_idx in reversed(range(1, xs.shape[1])):
            dec_h = self.decorder(xs[:, time_idx])
            out = self.hidden_to_output(dec_h)
            ys.append(out.data)
            self.loss += (xs[:, time_idx-1] - out)**2
        return ys

    def predict(self, xs, ys, out):
        for time_idx in reversed(range(1, xs.shape[1])):
            dec_h = self.decorder(out)
            out = self.hidden_to_output(dec_h)
            ys.append(out.data)
        return ys

    def __call__(self, xs):
        n_batch, n_times, dim_obs = xs.shape
        for time_idx in [0, 1]:  # range(n_times):
            x = xs[:, time_idx]
            # h = self.input_to_hidden(x)
            hidden = self.encorder(x)
            # print("hidden", hidden)

        self.set_decorder_init_state()
        ys = []
        out = self.hidden_to_output(self.decorder.h)

        ys.append(out)
        self.loss = (xs[:, -1] - out)**2

        chainer.report({'loss': self.loss}, self)

        if self.is_train:
            self.out = self.train(xs, ys)
        else:
            self.out = self.predict(xs, ys, out)
