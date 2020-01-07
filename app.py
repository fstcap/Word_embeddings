import os
import sys
import argparse
import collections
import random
import math
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

start_time = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
print('start_time', start_time)

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument(
    '--output',
    type=str,
    default=os.path.join(current_path,'output'))

FLAGS,unparsed = parser.parse_known_args()

print("FLAGS",FLAGS)
if not os.path.exists(FLAGS.output):
    os.makedirs(FLAGS.output)

#将数据读入字符串列表。
def read_data(filename):
    """读取数据单词列表。"""
    with open(filename,'r') as f:
        data = tf.compat.as_str(f.read()).split()
    return data

#Step 1: 获取数据
vocabulary = read_data(os.path.join(current_path,'input','text8'))
print("\033[0;32mData size",len(vocabulary),'Data_type',type(vocabulary),'Data[0:5]',vocabulary[0:5],"\033[0m")

#Step 2: 构建字典并用UNK令牌替换罕见的单词
vocabulary_size = 10000

def build_dataset(words, n_words):
    """将原始输入处理为数据集。"""
    count = [['UNK',-1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word,_ in count:
        dictionary[word] = len(dictionary)
    print("\033[0;33mdictionary[UNK]=\033[0m",dictionary['UNK'])
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word,0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    print('count_0_5',count[0:5])
    print('data_0_10',data[:10],'\ndata_len',len(data),'\ndata_count_0',data.count(0))
    print('reversed_dictionary_type',type(reversed_dictionary),'\nreversed_dictionary_0_5',[reversed_dictionary[i] for i in data[:10]])
    return data, count, dictionary, reversed_dictionary

#	填写4个全局变量:
#	data - 代码列表（从0到vocabulary_size-1的整数）。
#		这是原始文本，但单词被其代码替换
#	count - 单词（字符串）到出现次数的映射
#	dictionary - 单词（字符串）到其代码的映射（整数）
#	reverse_dictionary - 将代码（整数）映射到单词（字符串）
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,vocabulary_size)
del vocabulary #减少记忆内存

#Step 3: 用于为skip-gram模型生成训练批次的函数。
data_index = 0
def generate_batch(batch_size,num_skips,skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size),dtype=np.int32)
    labels = np.ndarray(shape=(batch_size,1),dtype=np.int32)
    span = 2 * skip_window + 1 # 窗口大小
    buffer = collections.deque(maxlen=span)
    # 如果窗口最后一位字或词的索引值超过数据长度，重置data_index
    if data_index + span  > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index+span]) # 在窗口中放入第一组字或词
    data_index += span # 将data_index的位置窗口外右边的第一个位置
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window] # 初始化窗口中上下文的索引，除了中间字或词的索引
        words_to_use = random.sample(context_words,num_skips) # 从窗口中上下文索引随机获取num_skips个窗口中字或词的索引值
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window] # 将窗口中中间索引值赋值给batch
            labels[i * num_skips + j,0] = buffer[context_word] # 窗口中中间字或词的索引的上下文赋值给labels
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index]) # 将窗口外右边的第一个位置的字或词的索引从后面塞进窗口，窗口的第一个索引就会自动益出
            data_index += 1
    # 回溯一点，以避免在批处理结束时跳过单词
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print("\033[0;33m",batch[i], reverse_dictionary[batch[i]],"\033[0m", '->',labels[i,0], reverse_dictionary[labels[i,0]])

# Step 4: 创建和训练skip-gram model
batch_size = 128
embedding_size = 128 # 嵌入向量的维数。
skip_window = 1 # 左右要考虑多少个单词。
num_skips = 2 # 重复使用输入生成标签的次数。
num_sampled = 64 # 要抽样的负面例子数量。

"""
我们选择一个随机验证集来对最近邻居进行抽样。
在这里，我们将验证样本限制为具有低数字ID的单词，按构造也是最常见的。
这3个变量仅用于显示模型精度，它们不影响计算。
"""
valid_window = 200 # 只选择分布头部的开发样本。
x_valid, y_valid = generate_batch(valid_window,num_skips,skip_window)
data_index = 0
EPOCHS = 100

X_train, Y_train = [], []
for epoch in range(EPOCHS):
    x, y = generate_batch(batch_size,num_skips,skip_window)
    X_train.append(x)
    Y_train.append(y)

dataset = tf.data.Dataset.from_tensor_slices((X_train,Y_train))

def transale_label(features,labels):
    #labels = tf.one_hot(indices=labels,depth=vocabulary_size,dtype=tf.float32)
    labels = tf.dtypes.cast(labels,dtype=tf.int64)
    return features, labels
train_dataset = dataset.map(transale_label)

for x, y in train_dataset:
    sampled_values = tf.random.log_uniform_candidate_sampler(
        true_classes=y,
        num_true=1,
        num_sampled=num_sampled,
        unique=True,
        range_max=vocabulary_size,
        seed=None)
    tf.print(f"\033[1;35mSampled_values:\033[0m{sampled_values}")

# 自定义embedding层
class LookupEmbedding(tf.keras.layers.Layer):
    def __init__(self,output_dim,**kwargs):
        super(LookupEmbedding,self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = tf.nn.relu
    def build(self,input_shape):
        self.kernel = tf.Variable(
            tf.random.uniform(
                shape=(vocabulary_size,self.output_dim),
                minval=-1.0,
                maxval=1.0))
    def call(self,x):
        x = tf.dtypes.cast(x,dtype=tf.int32)
        #print("\n\033[0;32mx_shape:\033[0m",tf.shape(x),"\n\033[0;32mdata_index:\033[0m",data_index)
        embedding = tf.nn.embedding_lookup(params=self.kernel,ids=x) 
        return self.activation(features=embedding)
    def compute_output_shape(self,input_shape):
        return (vocabulary_size,self.output_dim)
    def get_config(self):
        config = {'activation': self.activation}
        base_config = super(Activation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DenseLayer(tf.keras.layers.Layer):
    def __init__(self,output_dim,activation=False,**kwargs):
        super(DenseLayer,self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activation
    def build(self,input_shape):
        self.kernel = tf.Variable(
            tf.random.uniform(
                shape=(input_shape[1],self.output_dim),
                minval=-1,
                maxval=1,
                seed=234),name="kernel")
        self.biases = tf.Variable(
            tf.zeros(
                shape=(1,self.output_dim)),name="bias")
    def call(self,x):
        matmul = tf.linalg.matmul(x,self.kernel) + self.biases
        if self.activation:
            return self.activation(matmul)
        else:
            return matmul
    def compute_output_shape(self,input_shape):
        return (input_shape[1],self.output_dim)
    def get_config(self):
        config = {}
        if self.activation:config = {'activation':self.activation}
        base_config = super(Activation, self).get_config()
        return dict(list(base_config.items()) + list(config.items())) 

class CreateModel(tf.keras.Model):
    def __init__(self):
        super(CreateModel,self).__init__()
        self.lueb = LookupEmbedding(embedding_size,input_shape=(1,))
        self.d1 = DenseLayer(1,activation=tf.nn.sigmoid)
    def call(self, x):
        x = self.lueb(x)
        return self.d1(x)

# 自定义loss函数
def loss_fn(y_true, y_pred):
    sub = tf.math.subtract(y_pred, y_true)
    losses = tf.math.abs(sub)
    #losses = tf.math.sqrt(losses)
    #tf.print(f"\033[1;35mlosses:\033[0m{losses}")
    loss = tf.math.reduce_mean(losses)
    return loss
    
accuracy_train = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy_train')
model = CreateModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


#@tf.function
def step_train(inputs,targets):
    with tf.GradientTape() as tape:
        y_pred = model(inputs) 
        loss = loss_fn(targets,y_pred)
    grads = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    accuracy_train(targets,y_pred)
    return loss

train_loss_results, train_accuracy_results = [], []
test_loss_results, test_accuracy_results = [], []
num_epochs = 101

for epoch in range(num_epochs):
    total_loss = 0.0
    num_batchs = 0
    for x, y in train_dataset:
        total_loss += step_train(x, y)
        num_batchs += 1
    train_loss_results.append(total_loss/num_batchs)
    train_accuracy_results.append(accuracy_train.result())
    if epoch % 1 == 0:
        print(
            f"\033[1;35mEpoch:\033[0m{epoch}",
            f"\033[1;35mLoss:\033[0m{train_loss_results[-1]}",
            f"\033[1;35mAccuracy:\033[0m{train_accuracy_results[-1]}")

fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')
axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(
    train_loss_results, 'r',
    test_loss_results, 'b')
axes[0].legend(['train_loss', 'test_loss'])

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(
    train_accuracy_results, 'r',
    test_accuracy_results, 'b')
axes[1].legend(['train_accuracy', 'test_accuracy'])

plt.savefig(os.path.join(FLAGS.output,"train_steps.png"))

emb_layer = model.layers[0]
weights = emb_layer.get_weights()[0]

# 为embeddings编写相应的标签。
with open(os.path.join(FLAGS.output,'meta.tsv'), 'w') as f:
    for i in range(vocabulary_size):
        f.write(reverse_dictionary[i] + '\n')
    
with open(os.path.join(FLAGS.output,'vecs.tsv'), 'w') as f:
    for value in weights:
        f.write('\t'.join([str(x) for x in value]) + "\n")
