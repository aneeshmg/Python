import json
import torch
import bcolz
import np
import pickle
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


dev_data_file = './data/dev.jsonl'
training_data_file = './data/train.jsonl'
test_data_file = './data/test.jsonl'
glove_embeddings_file = './data/glove.6B.50d.txt'

# Parse and load the files to memory
dev_data = []
with open(dev_data_file) as f:
    for line in f:
        dev_data.append(json.loads(line))

training_data = []
with open(training_data_file) as f:
    for line in f:
        training_data.append(json.loads(line))

test_data = []
with open(test_data_file) as f:
    for line in f:
        test_data.append(json.loads(line))


words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir=f'./data/6B.50.dat', mode='w')

with open(glove_embeddings_file, 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)

vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=f'./data/6B.50.dat', mode='w')
vectors.flush()
pickle.dump(words, open(f'./data/6B.50_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'./data/6B.50_idx.pkl', 'wb'))

vectors = bcolz.open('./data/6B.50.dat')[:]
words = pickle.load(open('./data/6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open('./data/6B.50_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}
emb_dim = 50

# data = dev_data
data = training_data


# Embeddings
premise_embedding_array = []
hypothesis_embedding_array = []
premise_embedding_array_test = []
hypothesis_embedding_array_test = []



for i in range(len(data)):
    premise = data[i]['sentence1'].split(' ')
    premise.extend(["<PAD>" for i in range(15 - len(premise))])
    hypothesis = data[i]['sentence2'].split(' ')
    hypothesis.extend(["<PAD>" for i in range(15 - len(hypothesis))])
    labels = data[i]['gold_label']
    premise_embedding = np.zeros((15, emb_dim))
    hypothesis_embedding = np.zeros((15, emb_dim))

    for j, word in enumerate(premise):
        try: 
            premise_embedding[j] = glove[word]
        except KeyError:
            premise_embedding[j] = np.ones(emb_dim) * np.random.sample()
    
    premise_embedding_array.append(torch.from_numpy(premise_embedding))

    for j, word in enumerate(hypothesis):
        try: 
            hypothesis_embedding[j] = glove[word]
        except KeyError:
            hypothesis_embedding[j] = np.ones(emb_dim) * np.random.sample()

    hypothesis_embedding_array.append(torch.from_numpy(hypothesis_embedding))


for i in range(len(test_data)):
    premise = data[i]['sentence1'].split(' ')
    premise.extend(["<PAD>" for i in range(15 - len(premise))])
    hypothesis = data[i]['sentence2'].split(' ')
    hypothesis.extend(["<PAD>" for i in range(15 - len(hypothesis))])
    labels = data[i]['gold_label']
    premise_embedding = np.zeros((15, emb_dim))
    hypothesis_embedding = np.zeros((15, emb_dim))

    for j, word in enumerate(premise):
        try: 
            premise_embedding[j] = glove[word]
        except KeyError:
            premise_embedding[j] = np.ones(emb_dim) * np.random.sample()
    
    premise_embedding_array_test.append(torch.from_numpy(premise_embedding))

    for j, word in enumerate(hypothesis):
        try: 
            hypothesis_embedding[j] = glove[word]
        except KeyError:
            hypothesis_embedding[j] = np.ones(emb_dim) * np.random.sample()

    hypothesis_embedding_array_test.append(torch.from_numpy(hypothesis_embedding))



multiplier = np.double(torch.ones([30,1]))

labels_embedding = { 
    'entailment': torch.from_numpy(multiplier * np.double([0.0, 0.0, 1.0])),
    'contradiction' : torch.from_numpy(multiplier * np.double([0.0, 1.0, 0.0])),
    'neutral' : torch.from_numpy(multiplier * np.double([1.0, 0.0, 0.0])),
    'error' : torch.from_numpy(multiplier * np.double([0, 0, 0]))
}

class AssignmentNN(nn.Module):
    def __init__(self):
        super(AssignmentNN, self).__init__()
        input_one = 50
        hidden_one = 500
        output_one = 300 # Reporting results for 100 and 300
        input_two = 300
        hidden_two = 50 # Reporting results for 100 and 50
        output_two = 3

        self.f = nn.Linear(input_one, hidden_one)
        self.relu = nn.ReLU()
        self.f_h = nn.Linear(hidden_one, output_one)

        self.g = nn.Linear(input_two, hidden_two)
        self.tanh = nn.Tanh()
        self.g_h = nn.Linear(hidden_two, output_two)


    def forward(self, p, h):
        out_p = self.f(p)
        out_p = self.relu(out_p)
        out_p = self.f_h(out_p)
        out_p = self.relu(out_p)

        out_h = self.f(h)
        out_h = self.relu(out_h)
        out_h = self.f_h(out_h)
        out_h = self.relu(out_h)

        out = torch.cat((out_p, out_h))

        out = self.g(out)
        out = self.tanh(out)
        out = self.g_h(out)
        out = self.tanh(out)
        
        
        return F.log_softmax(out, dim = 1)


model = AssignmentNN()

# Use Double instead of float
model = model.double()

# Cross Entropy Loss 
error = nn.L1Loss()

# SGD Optimizer
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 3
count = 0
loss_list = []
iteration_list = []
accuracy_list = []

for epoch in range(num_epochs):

    for i in range(len(data)):

        premise = Variable(premise_embedding_array[i])
        hypothesis = Variable(hypothesis_embedding_array[i])
        labels= data[i]['gold_label']
        labels = Variable(labels_embedding[labels])

        optimizer.zero_grad()

        outputs = model(premise, hypothesis)

        loss = error(outputs, labels)

        loss.backward()

        optimizer.step()

        count += 1
        
        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Predict test dataset
            for j in range(len(test_data)):

                premise_test = Variable(premise_embedding_array_test[j])
                hypothesis_test = Variable(hypothesis_embedding_array_test[j])
                labels_test = test_data[j]['gold_label']
                try:
                    labels_test = Variable(labels_embedding[labels_test])
                except Exception:
                    labels_test = Variable(labels_embedding['error'])
                
                # Forward propagation
                outputs = model(premise_test, hypothesis_test)
                
                # Get predictions from the maximum value
                predicted = outputs.data
                
                # Total number of labels
                total += len(labels_test)

                # Total correct predictions
                correct += (predicted == labels_test).sum()

            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            if count % 1000 == 0:
                # Print Loss
                print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data[0], accuracy))
