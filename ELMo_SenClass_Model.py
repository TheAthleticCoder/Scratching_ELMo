
# coding: utf-8

# In[1]:


#Importing Libraries
import pandas as pd
import numpy as np
import re


# In[2]:


#Initializing torch and cuda device
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device


# In[3]:


import torch
import torchtext

print("Torch Text Version : {}".format(torchtext.__version__))


# In[4]:


#Loading the datasets
df_train = pd.read_csv('yelp-subset.train.csv')
df_valid = pd.read_csv('yelp-subset.dev.csv')
df_test = pd.read_csv('yelp-subset.test.csv')
df_test.head()


# In[5]:


#Custom Tokenizer (Rejected due to poor performance)
def my_tokenizer(s):
    #make everythign small
    s = s.lower()
    #regex replace repeated punctuations
    s = re.sub(r'([!?.]){2,}', r'\1', s)
    #regex replace ' and " with 
    s = re.sub(r'([\'\"])',"", s)
    s = re.sub(r'\\', "", s)
    s = re.sub(r'\\[a-z]', "", s)
    s = re.sub(r'/', " / ", s)
    s = re.sub(r'&', " & ", s)
    #regex add space before punctuations
    s = re.sub(r'([!?.])', r' \1', s)
    #replace urls with a special token
    s = re.sub(r'((http|https)://[^\s]+)', r' <url> ', s)
    #replace emails with a special token
    s = re.sub(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', r' <email> ', s)
    #replace phone numbers with a special token
    s = re.sub(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})', r' <phone> ', s)
    #replace money with a special token
    s = re.sub(r'(\$[0-9]+)', r' <money> ', s)
    #replace time with a special token
    s = re.sub(r'([0-9]+:[0-9]+)', r' <time> ', s)
    #replace dates with a special token
    s = re.sub(r'([0-9]+/[0-9]+/[0-9]+)', r' <date> ', s)
    #replace numbers with a special token
    s = re.sub(r'([0-9]+)', r' <num> ', s)
    return s.split()


# In[6]:


#Importing tokenizer from torch.data
import torchtext
from torchtext.data import get_tokenizer
tokenizer = get_tokenizer('basic_english')

sentences = df_train['text'].values
tokenized = [tokenizer(sent) for sent in sentences]
val_sentences = df_valid['text'].values
val_tokenized = [tokenizer(sent) for sent in val_sentences]
tokenized = [tokenizer(sent) for sent in sentences]
test_sentences = df_test['text'].values
test_tokenized = [tokenizer(sent) for sent in test_sentences]
print(test_tokenized[0])


# In[7]:


#### Max Length
# max_len = max([len(sent) for sent in tokenized])
# max_len
# max_len = 0
# longest_sentence_num = 0
# for i in range(len(tokenized)):
#     if len(tokenized[i]) > max_len:
#         max_len = len(tokenized[i])
#         longest_sentence_num = i
# print(tokenized[longest_sentence_num])
# print(longest_sentence_num)


# In[8]:


#Importing Dataset and DataLoader
from torch.utils.data import Dataset, DataLoader

#Creatin vocabulary
def create_vocab(tokenized):
    vocab = {}
    freq = {}
    #add <PAD> and <UNK> tokens
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    freq['<PAD>'] = 0
    freq['<UNK>'] = 0
    #add tokens from tokenized sentences to vocab and freq
    for sent in tokenized:
        for word in sent:
            if word not in vocab:
                vocab[word] = len(vocab)
                freq[word] = 1
            else:
                freq[word] += 1
    #words with freq less than 5 are replaced with <UNK> token
    vocab_final = {}
    vocab_final['<PAD>'] = 0
    vocab_final['<UNK>'] = 1
    #add tokens from tokenized sentences to vocab_final if freq is greater than 5
    for word in vocab:
        if freq[word] >= 2:
            vocab_final[word] = len(vocab_final)
    return vocab_final

#build vocab from tokenized sentences
vocab = create_vocab(tokenized)
print(list(vocab.items())[:10])
#print length of vocab
print(len(vocab))


# In[9]:


#Changing tokens in tokenized sentences to indices
def token2index_dataset(tokenized):
    indices = []
    for sent in tokenized:
        index = []
        for word in sent:
            if word in vocab:
                index.append(vocab[word])
            else:
                index.append(vocab['<UNK>'])
        indices.append(index)
    return indices
train_data = token2index_dataset(tokenized)
valid_data = token2index_dataset(val_tokenized)
test_data = token2index_dataset(test_tokenized)
#print first 4 sentences
print(test_data[:4])


# In[10]:


#Function to pad sentences to max length
def pad_sents(sents, pad_token, max_len):
    padded_sents = []
    for sent in sents:
        if len(sent) < max_len:
            padded_sents.append(sent + [pad_token] * (max_len - len(sent)))
        else:
            padded_sents.append(sent[:max_len])
    return padded_sents
train_labels = df_train['label'].values
valid_labels = df_valid['label'].values
test_labels = df_test['label'].values
print(len(test_labels))
print(len(test_data))


# In[11]:


#Constructing Dataset
class ELMoDataset(Dataset):
    def __init__(self, data, labels, max_len):
        self.data = data
        #pad sentences to max length
        self.data = pad_sents(self.data, vocab['<PAD>'], max_len)
        #all sentences till max_len-1
        self.back_data = [sent[:-1] for sent in self.data]
        #all sentences from 1 to max_len
        self.forward_data = [sent[1:] for sent in self.data]
        self.input_data = self.data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        #converting to torch tensors
        back_data = torch.tensor(self.back_data[idx])
        forward_data = torch.tensor(self.forward_data[idx])
        input_data = torch.tensor(self.input_data[idx])
        labels = torch.tensor(self.labels[idx])
        return back_data, forward_data, input_data, labels

max_len = 250 #setting the length to a fixed value
train_dataset = ELMoDataset(train_data, train_labels, max_len)
valid_dataset = ELMoDataset(valid_data, valid_labels, max_len)
test_dataset = ELMoDataset(test_data, test_labels, max_len)
print(test_dataset[0])


# In[12]:


#Display Dataset dimensions
print(len(train_dataset))
print(len(valid_dataset))
print(len(test_dataset))


# In[13]:


#Create dataloaders and organize datasets based on batch size
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
forward_train, back_train, input_train, labels_train = next(iter(train_loader))
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
forward_valid, back_valid, input_valid, labels_valid = next(iter(valid_loader))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
forward_test, back_test, input_test, labels_test = next(iter(test_loader))
print('Train Dataset:')
print(forward_train.shape)
print(back_train.shape)
print(input_train.shape)
print(labels_train.shape)
print('Valid Dataset:')
print(forward_valid.shape)
print(back_valid.shape)
print(input_valid.shape)
print(labels_valid.shape)
print('Test Dataset:')
print(forward_test.shape)
print(back_test.shape)
print(input_test.shape)
print(labels_test.shape)


# In[14]:


#Loading GloVe vectors
from torchtext.vocab import GloVe
glove = GloVe(name='6B', dim=100)


# In[15]:


def create_embedding_matrix(vocab, embedding_dim):
    embedding_matrix = torch.zeros((len(vocab), embedding_dim))
    for word, index in vocab.items():
        if word in glove.stoi:
            embedding_matrix[index] = glove.vectors[glove.stoi[word]]
    return embedding_matrix.detach().clone()

#initialize embedding matrix
embedding_matrix = create_embedding_matrix(vocab, 100)
print(embedding_matrix.shape)
print(embedding_matrix[0])


# In[16]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[17]:


#Constructing Elmo Dataset
class ELMo(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size, max_len, embedding_matrix):
        super(ELMo, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.max_len = max_len
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.embedding.weight = nn.Parameter(self.embedding.weight, requires_grad=True)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim*2, hidden_dim, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim*2, vocab_size)
    def forward(self,back_data):
        back_embed = self.embedding(back_data)
        back_lstm1, _ = self.lstm1(back_embed)
        #back_lstm1 is shape of (batch_size, max_len, hidden_dim*2)
        back_lstm2, _ = self.lstm2(back_lstm1)
        linear_out = self.linear_out(back_lstm2)
        return linear_out
    
#initialize elmo
elmo = ELMo(len(vocab), 100, 100, batch_size, max_len, embedding_matrix)
print(elmo)


# In[19]:


elmo.to(device)
#Initializing optimizer
optimizer = optim.Adam(elmo.parameters(), lr=0.001)
#Initializing loss function
criterion = nn.CrossEntropyLoss()


# In[20]:


#Training elmo
elmo_losses = {'train': [], 'val': [], 'epoch' : []}
epochs = 4
lowest_val_loss = 100
for epoch in range(epochs):
    train_loss = 0
    elmo.train()
    for i, (forward_data, back_data, input_data, labels) in enumerate(train_loader):
        forward_data = forward_data.to(device)
        back_data = back_data.to(device)
        optimizer.zero_grad()
        output = elmo(back_data)
        output = output.view(-1, len(vocab))
        target = forward_data.view(-1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if i%500 == 0:
            print('Epoch: {}/{}'.format(epoch+1, epochs), 'Step: {}'.format(i), 'Loss: {}'.format(loss.item()), 'Train Loss: {}'.format(train_loss/(i+1)))
    valid_loss = 0
    elmo.eval()
    for i, (forward_val, back_val, input_val, labels_val) in enumerate(valid_loader):
        forward_val = forward_val.to(device)
        back_val = back_val.to(device)
        output_val = elmo(back_val)
        output_val = output_val.view(-1, len(vocab))
        target_val = forward_val.view(-1)
        loss_val = criterion(output_val, target_val)
        valid_loss += loss_val.item()
        if i%500 == 0:
            print('Epoch: {}/{}'.format(epoch+1, epochs), 'Step: {}'.format(i), 'Loss: {}'.format(loss_val.item()), 'Valid Loss: {}'.format(valid_loss/(i+1)))
    print('Epoch: {}/{}'.format(epoch+1, epochs), 'Train Loss: {}'.format(train_loss/(i+1)), 'Valid Loss: {}'.format(valid_loss/(i+1)))
    elmo_losses['train'].append(train_loss/len(train_loader))
    elmo_losses['val'].append(valid_loss/len(valid_loader))
    elmo_losses['epoch'].append(epoch)
    if valid_loss < lowest_val_loss:
        lowest_val_loss = valid_loss
        torch.save(elmo.state_dict(), 'elmo.pt')
        print('Model updated and saved')


# In[21]:


elmo.load_state_dict(torch.load('elmo.pt'))


# In[22]:


print(elmo_losses)


# In[23]:


get_ipython().magic('matplotlib notebook')
from matplotlib import pyplot as plt


# In[24]:


plt.plot(elmo_losses['epoch'], elmo_losses['train'], label='Train Loss')
plt.plot(elmo_losses['epoch'], elmo_losses['val'], label='Validation Loss')
plt.title('ELMo Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('ELMo.png')


# In[25]:


for name, param in elmo.named_parameters():
    if param.requires_grad:
        print(name, param.data, param.shape)


# In[26]:


elmo_embeddings = list(elmo.parameters())[0].cpu().detach().numpy()


# In[27]:


torch.save(elmo_embeddings, 'elmo_embeddings.pt')


# In[28]:


word = '<UNK>'
word_index = vocab[word]
elmo_embeddings[word_index]


# In[29]:


elmo_lstm1 = elmo.lstm1
print(elmo_lstm1.parameters())
elmo_lstm2 = elmo.lstm2
print(elmo_lstm2.parameters())


# In[30]:


elmo_embeddings = list(elmo.parameters())[0].to(device)


# In[31]:


class SenClass(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size, max_len, embedding_matrix, elmo_embeddings, elmo_lstm1, elmo_lstm2, num_classes):
        super(SenClass, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.max_len = max_len
        self.embeddings = nn.Embedding.from_pretrained(elmo_embeddings)
        self.embeddings.weight = nn.Parameter(self.embeddings.weight, requires_grad=True)
        self.weights = nn.Parameter(torch.tensor([0.33, 0.33, 0.33]), requires_grad=True)
#         self.weights = nn.Parameter(torch.tensor([0.2, 0.2, 0.2]))
        self.lstm1_ft = elmo_lstm1
        self.lstm2_ft = elmo_lstm2
        self.linear1 = nn.Linear(embedding_dim, hidden_dim*2)
        self.linear2 = nn.Linear(hidden_dim*2, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_data):
        embeds = self.embeddings(input_data)
        embeds_change = self.linear1(embeds)
        hidden1, _ = self.lstm1_ft(embeds)
        hidden2, _ = self.lstm2_ft(hidden1)
        elmo_embed = (self.weights[0]*hidden1 + self.weights[1]*hidden2 + self.weights[2]*embeds_change)/(self.weights[0]+self.weights[1]+self.weights[2])
#         elmo_embed = (0.33*hidden1 + 0.33*hidden2 + 0.33*embeds_change)/(1)
        elmo_embed_max = torch.max(elmo_embed, dim=1)[0]
        elmo_embed_max_drop = self.dropout(elmo_embed_max)
        linear_out = self.linear2(elmo_embed_max_drop)
        return linear_out

sen_class = SenClass(len(vocab), 100, 100, batch_size, max_len, embedding_matrix, elmo_embeddings, elmo_lstm1, elmo_lstm2, 5)
print(sen_class)


# In[32]:


# elmo = elmo.to(device)
sen_class.to(device)
#Initializing optimizer
optimizer = optim.Adam(sen_class.parameters(), lr=0.001)
#Initializing loss function
criterion = nn.CrossEntropyLoss()


# In[33]:


from sklearn.metrics import f1_score, confusion_matrix, accuracy_score


# In[34]:


sen_class_losses = {'train': [], 'val': [], 'epoch': []}
sen_class_acc = {'train': [], 'val': [], 'epoch': []}
sen_class_f1 = {'train': [], 'val': [], 'epoch': []}
sc_lowest_val_loss = 1000
epochs = 5
for epoch in range(epochs):
    train_loss = 0
    train_acc = 0
    train_f1 = 0
    sen_class.train()
    for i, (forward_data, back_data, input_data, labels) in enumerate(train_loader):
        input_data = input_data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = sen_class(input_data)
        loss = criterion(output, labels)
        loss.backward()
        #update weights
        optimizer.step()
        #add loss to train_los
        train_loss += loss.item()
        _, preds = torch.max(output, 1)
        #get accuracy
        train_acc += accuracy_score(labels.cpu(), preds.cpu())
        #get micro f1 score
        train_f1 += f1_score(labels.data.cpu().numpy(), preds.cpu().numpy(), average='micro')
        #get confusion matrix
        if i%500 == 0:
            print('Epoch: {}/{}'.format(epoch+1, epochs), 'Step: {}'.format(i), 'Loss: {}'.format(loss.item()), 'Train Loss: {}'.format(train_loss/(i+1)))
    valid_loss = 0
    valid_acc = 0
    valid_f1 = 0
    sen_class.eval()
    for i, (forward_val, back_val, input_val, labels_val) in enumerate(valid_loader):
        input_val = input_val.to(device)
        labels_val = labels_val.to(device)
        outputs_val = sen_class(input_val)
        loss_val = criterion(outputs_val, labels_val)
        valid_loss += loss_val.item()
        #get predictions
        _, preds = torch.max(outputs_val, 1)
        #get accuracy
        valid_acc += accuracy_score(labels_val.cpu(), preds.cpu())
        #get micro f1 score
        valid_f1 += f1_score(labels_val.data.cpu().numpy(), preds.cpu().numpy(), average='micro')
        if 1%500 == 0:
            print('Epoch: {}/{}'.format(epoch+1, epochs), 'Step: {}'.format(i+1), 'Loss: {}'.format(loss.item()), 'Valid Loss: {}'.format(valid_loss/(i+1)))
    print('Epoch: {}/{}'.format(epoch+1, epochs), 'Train Loss: {}'.format(train_loss/(i+1)), 'Valid Loss: {}'.format(valid_loss/(i+1)))
    if valid_loss < sc_lowest_val_loss:
        sc_lowest_val_loss = valid_loss
        torch.save(sen_class.state_dict(), 'sen_class.pt')
        print('Clasification Model updated and saved')
    sen_class_losses['train'].append(train_loss/len(train_loader))
    sen_class_acc['train'].append(train_acc/len(train_loader))
    sen_class_f1['train'].append(train_f1/len(train_loader))
    sen_class_losses['epoch'].append(epoch+1)
    sen_class_acc['epoch'].append(epoch+1)
    sen_class_f1['epoch'].append(epoch+1)
    sen_class_losses['val'].append(valid_loss/len(valid_loader))
    sen_class_acc['val'].append(valid_acc/len(valid_loader))
    sen_class_f1['val'].append(valid_f1/len(valid_loader))


# In[35]:


#Printing the dictionary of accuracies
print('Losses:')
print(sen_class_losses)
#printing the dictionary of accuracies
print('Accuracy:')
print(sen_class_acc)
#printign the dictionary of f1 scores
print('Micro F1:')
print(sen_class_f1)


# In[36]:


plt.plot(sen_class_losses['epoch'], sen_class_losses['train'], label='Train Loss')
plt.plot(sen_class_losses['epoch'], sen_class_losses['val'], label='Validation Loss')
plt.title('SenClass Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('SenClass.png')


# In[37]:


plt.plot(sen_class_acc['epoch'], sen_class_acc['train'], label='Train Accuracy')
plt.plot(sen_class_acc['epoch'], sen_class_acc['val'], label='Validation Accuracy')
plt.title('SenClass Model')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('SenClassAccuracy.png')


# In[38]:


sen_class.load_state_dict(torch.load('sen_class.pt'))


# In[39]:


for name, param in sen_class.named_parameters():
    if param.requires_grad:
        print(name, param.data, param.shape)


# In[40]:


#Predictions on the Test Data
sen_class.load_state_dict(torch.load('sen_class.pt'))
sen_class.eval()
test_preds = []
for i, (forward_data, back_data, input_data, labels) in enumerate(test_loader):
    input_data = input_data.to(device)
    labels = labels.to(device)
    outputs = sen_class(input_data)
    #get predictions
    _, preds = torch.max(outputs, 1)
    test_preds.extend(preds.cpu().numpy())
#calculate micro f1 score
test_f1 = f1_score(test_labels, test_preds, average='micro')
print("F1 Micro:")
print(test_f1)
#calculate test accuracy
print("Accuracy:")
test_acc = accuracy_score(test_labels, test_preds)
print(test_acc)
#calculate confusion matrix
print("Confusion Matrix:")
test_confusion = confusion_matrix(test_labels, test_preds)
print(test_confusion)

