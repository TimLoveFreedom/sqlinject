
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


#read the samples
with open('positive.txt', 'r',encoding='utf-8') as f:
    data_positive = f.read()
with open('negative.txt', 'r',encoding='utf-8') as f:
    data_negative = f.read()


# In[3]:


# initial the labels
label_positive=np.ones(len(data_positive))
label_negative=np.zeros(len(data_negative))


# In[4]:


#the char of train
charList=['\x00','\x0b','\x0c','\x19','\t',' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
vocab = sorted(charList, key=lambda x:x)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}


# In[5]:


#ints of positive
ints_positive=[]
for index,each in enumerate(data_positive.split('\n')):
    ints_positive.append([vocab_to_int[word] for word in each.rstrip()])
#ints of negative
ints_negative=[]
for each in data_negative.split('\n'):
    ints_negative.append([vocab_to_int[word] for word in each.rstrip()])


# In[6]:


#statistics of the positive
from collections import Counter
positive_lens = Counter([len(x) for x in ints_positive])
print("Zero-length positive: {}".format(positive_lens[0]))
print("Maximum positive length: {}".format(max(positive_lens)))

#statistics of the negative
negative_lens = Counter([len(x) for x in ints_negative])
print("Zero-length negative: {}".format(negative_lens[0]))
print("Maximum negative length: {}".format(max(negative_lens)))


# In[7]:


#delete zero-length
non_zero_idx_positive = [ii for ii, review in enumerate(ints_positive) if len(review) != 0]
print(len(non_zero_idx_positive))

non_zero_idx_negative = [ii for ii, review in enumerate(ints_negative) if len(review) != 0]
print(len(non_zero_idx_negative))


ints_positive = [ints_positive[ii] for ii in non_zero_idx_positive]
label_positive = np.array([label_positive[ii] for ii in non_zero_idx_positive])

ints_negative = [ints_negative[ii] for ii in non_zero_idx_negative]
label_negative = np.array([label_negative[ii] for ii in non_zero_idx_negative])


# In[8]:


# length of th sequence
seq_len=200
# ints_positive=[ip[:seq_len] for ip in ints_positive]
# ints_negative=[ip[:seq_len] for ip in ints_negative]
features_positive = np.zeros((len(ints_positive), seq_len), dtype=int)
for i, row in enumerate(ints_positive):
    features_positive[i, -len(row):] = np.array(row)[:seq_len]
features_negative = np.zeros((len(ints_negative), seq_len), dtype=int)
for i, row in enumerate(ints_negative):
    features_negative[i, -len(row):] = np.array(row)[:seq_len]


# In[9]:


#split the train set and test set  (undo)
# print(type(ints_positive))
# print(type(label_positive))
split_frac = 0.8
split_idx_positive = int(len(features_positive)*0.8)
split_idx_negative = int(len(features_negative)*0.8)
train_x, val_x =np.append(features_positive[:split_idx_positive],features_negative[:split_idx_negative],axis=0),np.append(features_positive[split_idx_positive:],features_negative[split_idx_negative:],axis=0)
train_y, val_y = np.append(label_positive[:split_idx_positive],label_negative[:split_idx_negative]),np.append(label_positive[split_idx_positive:],label_negative[split_idx_negative:])

test_idx = int(len(val_x)*0.5)
val_x, test_x = val_x[:test_idx], val_x[test_idx:]
val_y, test_y = val_y[:test_idx], val_y[test_idx:]


# In[ ]:


#shuffle
indices_train =np.arange(len(train_x))
# print(type(indices_train))
# print(indices_train)
np.random.shuffle(indices_train)  
for index,indice in enumerate(indices_train): 
    train_x[index]=train_x[indice]
    train_y[index]=train_y[indice]
indices_val = np.arange(len(val_x))
np.random.shuffle(indices_val)  
for index,indice in enumerate(indices_val): 
    val_x[index]=val_x[indice]
    val_y[index]=val_y[indice]
    test_x[index]=test_x[indice]
    test_y[index]=test_y[indice]
    


# In[17]:


print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))


# In[18]:


#super param
lstm_size = 256
lstm_layers = 2
batch_size = 1000
learning_rate = 0.001
n_words = len(vocab_to_int) + 1 # Adding 1 because we use 0's for padding, dictionary started at 1
epochs = 3


# In[19]:


# Create the graph object
graph = tf.Graph()
# Add nodes to the graph
with graph.as_default():
    inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
    labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')


# In[20]:


# Size of the embedding vectors (number of units in the embedding layer)
embed_size = 300 


# In[21]:


with graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs_)


# In[22]:


with graph.as_default():
    # Your basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    
    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    
    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)


# In[23]:


with graph.as_default():
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed,
                                             initial_state=initial_state)


# In[24]:


with graph.as_default():
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
    cost = tf.losses.mean_squared_error(labels_, predictions)
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# In[25]:


with graph.as_default():
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[26]:


def get_batches(x, y, batch_size=100):
    
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]


# In[27]:


with graph.as_default():
    saver = tf.train.Saver()
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(epochs):
        state = sess.run(initial_state)        
        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: 0.5,
                    initial_state: state}
            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)
            
            if iteration%5==0:
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss))

            if iteration%25==0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(val_x, val_y, batch_size):
                    feed = {inputs_: x,
                            labels_: y[:, None],
                            keep_prob: 1,
                            initial_state: val_state}
                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                print("Val acc: {:.3f}".format(np.mean(val_acc)))
            iteration +=1
    saver.save(sess, "checkpoints/sqlinject.ckpt")


# In[ ]:


test_acc = []
with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
        feed = {inputs_: x,
                labels_: y[:, None],
                keep_prob: 1,
                initial_state: test_state}
        batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
        test_acc.append(batch_acc)
    print("Test accuracy: {:.3f}".format(np.mean(test_acc)))


# In[ ]:




