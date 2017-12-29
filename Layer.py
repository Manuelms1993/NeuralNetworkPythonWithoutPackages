import numpy as np

class Layer():

    weights = []
    bias = []
    delta = []
    activation = []
    nNeurons = 0
    nInputs = 0


    def __init__(self, shape):

        def init_weights(shape):
            return np.random.randn(*shape) * 0.1

        self.nNeurons=shape[0]
        self.nInputs=shape[1]
        self.weights=init_weights(shape)
        self.bias=np.zeros((self.nNeurons,))
        self.delta=np.zeros((self.nNeurons,))
        self.activation=np.zeros((self.nNeurons,))