import numpy as np
import gzip
import cPickle
from Layer import Layer

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return .5 * (1 + np.tanh(0.5 * x))

def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h

def model(input,layer1,layer2,layer3):
    input = np.array([input])[0]
    layer1.activation = sigmoid(np.dot(input,np.transpose(layer1.weights)) + layer1.bias)
    layer2.activation = sigmoid(np.dot(input,np.transpose(layer2.weights)) + layer2.bias)
    layer3.activation = sigmoid(np.dot(layer2.activation,np.transpose(layer3.weights)) + layer3.bias)

def backPropagation (inputs,expectedOutput,layer1,layer2,layer3,lr=0.1):
    for i in range(inputs.shape[0]):
        layer3.delta = ((1-layer3.activation[i])*layer3.activation[i]) * (expectedOutput[i]-layer3.activation[i])
        layer2.delta = (layer2.activation[i]*(1-layer2.activation[i])) * np.dot(layer3.delta,layer3.weights)
        layer3.bias += lr*layer3.delta
        layer2.bias += lr*layer2.delta
        layer3.weights += lr* np.dot(np.transpose(np.array([layer3.delta])),np.array([layer2.activation[i]]))
        layer2.weights += lr* np.dot(np.transpose(np.array([layer2.delta])),np.array([inputs[i]]))

def MSE (expectedOutput,layer3):
    result = sum([((expectedOutput[i]-layer3.activation[i])**2) for i in range(layer3.nNeurons)])/layer3.nNeurons
    return sum(result)/expectedOutput.shape[0]

def testMean(teX,teY,layer1,layer2,layer3):
    result = 0
    t = len(teX)
    for s in range(0, t, 1):
        model(teX[s],layer1,layer2,layer3)
        expectedOutput = teY[s]
        result += 1 if np.argmax(expectedOutput) == np.argmax(layer3.activation) else 0
    return (float(result)/t)*100

def train():
    print "Loading data..."
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    trX, trY = train_set
    teX, teY = test_set
    f.close()
    print "-------"
    print trX.shape
    print trY.shape
    print teX.shape
    print teY.shape
    trY = one_hot(trY, 10)
    teY = one_hot(teY, 10)
    print "-------"
    layer1 = Layer((1,trX.shape[1]))
    layer2 = Layer((50,trX.shape[1]))
    layer3 = Layer((10,50))

    f = open("Cost.txt", 'w')
    f.write("Training cost"+"\n")
    f2 = open("Mean4.txt", 'w')
    f2.write("Hits Mean"+"\n")
    print "---------------------------------------"
    n = len(trX)
    for i in range(10):
        print "Iteration "+str(i)+":"
        cost = 0
        for start, end in zip(range(0, n, 128), range(128, n, 128)):
            model(trX[start:end],layer1,layer2,layer3)
            expectedOutput = trY[start:end]
            backPropagation(trX[start:end],expectedOutput,layer1,layer2,layer3)
            cost += MSE(expectedOutput,layer3)
        result = testMean(teX,teY,layer1,layer2,layer3)
        print "     Cost: "+str(cost)
        print "     Hit Mean: "+str(result)+" %"
        f.write(str(cost)+"\n")
        f2.write(str(result)+"\n")
        f.flush()
    f.close()

train()
import Graphics
Graphics.plotLearningCurve(["Cost.txt"],"Cost",True,1)
Graphics.plotLearningCurve(["Mean4.txt"],"Mean",True,0)