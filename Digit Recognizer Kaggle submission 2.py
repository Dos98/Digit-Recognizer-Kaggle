
# coding: utf-8

# In[7]:


# import all needed
import numpy
import scipy.special
import scipy.ndimage
import matplotlib.pyplot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


# neural network class definition
class neuralNetwork:
    
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        self.lr = learningrate
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    
    # train 
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass

    
    # query
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs


# In[9]:


# Inputs
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
# neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)


# In[10]:


# load 
training_data_file = open("/home/ayush/.kaggle/competitions/digit-recognizer/train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()


# In[26]:


# train the neural network

epochs = 5
for e in range(epochs):
    for record in training_data_list[1:]:
        all_values = record.split(',')
        inputs = ((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
        # rotated anticlockwise by 10 degrees
        inputs_plus10_img = scipy.ndimage.rotate(inputs.reshape(28,28), 10.0, cval=0.01, order=1, reshape=False)
        # rotated clockwise by 10 degrees
        inputs_minus10_img = scipy.ndimage.rotate(inputs.reshape(28,28), -10.0, cval=0.01, order=1, reshape=False)
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        n.train(inputs_plus10_img.reshape(1,784), targets)
        n.train(inputs_minus10_img.reshape(1,784), targets)
        pass
    pass


# In[27]:


# load the mnist test data CSV file into a list

test_data_file = open("/home/ayush/.kaggle/competitions/digit-recognizer/test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()


# In[28]:


# test the neural network

scorecard = []
for record in test_data_list[1:]:

    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[:]) / 255.0 * 0.99) + 0.01
    
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    scorecard.append(label)
    pass


# In[29]:


scorecard_array = numpy.asarray(scorecard)


# In[30]:


import pandas as pd
pd.DataFrame({"ImageId": list(range(1,len(scorecard)+1)), "Label": scorecard}).to_csv('submission.csv', index=False, header=True)

