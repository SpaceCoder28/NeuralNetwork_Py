import numpy as np #for matrix functionality

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
sigmoid = np.vectorize(sigmoid)
def dsigmoid(y):
  # return sigmoid(x) * (1 - sigmoid(x))
  return y * (1 - y)
dsigmoid = np.vectorize(dsigmoid)

class NeuralNetwork:
  def __init__(self, input_nodes, hidden_nodes, output_nodes):
    self.input_nodes = input_nodes
    self.hidden_nodes = hidden_nodes
    self.output_nodes = output_nodes
    
    #random initial weights between (-1,1)
    self.weights_ih = np.random.random((hidden_nodes, input_nodes))*2-1
    self.weights_ho = np.random.random((output_nodes, hidden_nodes))*2-1
    
    #biases for layers between (-1,1)
    self.bias_h = np.random.random((hidden_nodes, 1))*2-1
    self.bias_o = np.random.random((output_nodes, 1))*2-1
    
    self.learning_rate = 0.1
  
  def feedforward(self, input_array):
    inputs = np.matrix(input_array).transpose()
    
    #finding values for hidden layer
    hidden = np.matmul(self.weights_ih, inputs)
    hidden += self.bias_h #adding bias_h
    hidden = sigmoid(hidden)
    #finding values for output layer
    guess = np.matmul(self.weights_ho, hidden)
    guess += self.bias_o 
    guess = sigmoid(guess)
    
    #to return the data as list (input_array is also a list)
    return guess.tolist()
    
  def train(self, input_array, target_array):
    inputs = np.matrix(input_array).transpose()
    
    #finding values for hidden layer
    hidden = np.matmul(self.weights_ih, inputs)
    hidden += self.bias_h #adding bias_h
    hidden = sigmoid(hidden)
    
    #finding values for output layer
    outputs = np.matmul(self.weights_ho, hidden)
    outputs += self.bias_o 
    outputs = sigmoid(outputs)
    
    #convert target_array to column matrix
    targets = np.matrix(target_array).transpose()
    
    #calculating the error
    #ERROR = TARGETS - output
    output_errors = targets - outputs
    
    # let gradient = outputs * (1 - outputs);
    # Calculate gradient
    gradients = dsigmoid(outputs)
    gradients = np.multiply(gradients, output_errors)
    gradients *= self.learning_rate

    # Calculate deltas
    hidden_T = hidden.transpose()
    weight_ho_deltas = np.multiply(gradients, hidden_T)

    # Adjust the weights by deltas
    self.weights_ho = np.add(self.weights_ho, weight_ho_deltas)
    # Adjust the bias by its deltas (which is just the gradients)
    self.bias_o = np.add(self.bias_o, gradients)

    # Calculate the hidden layer errors
    who_t = self.weights_ho.transpose()
    hidden_errors = np.multiply(who_t, output_errors)

    # Calculate hidden gradient
    hidden_gradient = dsigmoid(hidden)
    hidden_gradient = np.multiply(hidden_gradient, hidden_errors)
    hidden_gradient = np.multiply(hidden_gradient, self.learning_rate)

    # Calcuate input->hidden deltas
    inputs_T = inputs.transpose()
    weight_ih_deltas = np.multiply(hidden_gradient, inputs_T);

    self.weights_ih = np.add(self.weights_ih, weight_ih_deltas)
    # Adjust the bias by its deltas (which is just the gradients)
    self.bias_h = np.add(self.bias_h, hidden_gradient)
    

#Expected feedforward functionality
nn = NeuralNetwork(2,1,1)
for i in range(0, 10000):
  p = np.random.rand()
  if (p < 0.25):
    nn.train([0, 0], [0])
  elif (p < 0.50):
    nn.train([0, 1], [1])
  elif (p < 0.75):
    nn.train([1, 0], [1])
  else:
    nn.train([1, 1], [0])

print "Training done"
a = 1
while(a != 0):
  array = input('Give an array: ')
  if (array == 0):
    a = 0
  break
  print nn.feedforward(array)
  
