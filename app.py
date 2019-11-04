from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import argparse
import collections
import numpy as np
import random
import math
import time

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

rightnow = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
print('rightnow', rightnow)

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(current_path,'log'),
    help='The log directory for TensorBoard summaries.')
parser.add_argument(
    '--output',
    type=str,
    default=os.path.join(current_path,'output'))

FLAGS,unparsed = parser.parse_known_args()

print("FLAGS",FLAGS)
#如果没有，则为TensorBoard变量or输出创建目录
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)
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
vocabulary_size = 50000

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

data_index = 0

#Step 3: 用于为skip-gram模型生成训练批次的函数。
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
            labels[i * num_skips + j, 0] = buffer[context_word] # 窗口中中间字或词的索引的上下文赋值给labels
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
valid_size = 12500

x_train, y_train = generate_batch(vocabulary_size + valid_size,num_skips,skip_window)
y_train = tf.one_hot(y_train,vocabulary_size).numpy()
print("\033[0;32mx_train shape:\033[0m",np.shape(x_train),"\n\033[0;32my_train shape:\033[0m",np.shape(y_train))

# 自定义embedding层
class LookupEmbedding(layers.Layer):
    def __init__(self,input_dim,output_dim,**kwargs):
        super(LookupEmbedding,self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
    def build(self,input_shape): 
        self.kernel = tf.Variable(
            tf.random.uniform(
                shape=(self.input_dim,self.output_dim),
                minval=-1.0,
                maxval=1.0))
        super(LookupEmbedding,self).build((self.input_dim,self.output_dim))
    def call(self,x):
        return tf.nn.embedding_lookup(params=self.kernel,ids=x)
    def compute_output_shape(self,input_shape):
        return (self.input_dim,self.output_dim)

# 自定义loss函数
def categorical_crossentropy(y_true, y_pred):
    #Y_true = tf.one_hot(tf.dtypes.cast(y_true,dtype=tf.int32),vocabulary_size)
    #print("\033[0;33my_true:\n",y_true,"\ny_pred:\n",y_pred)
    matmul = tf.linalg.matmul(y_true,tf.math.log(y_pred),transpose_b=True)
    return -tf.math.reduce_mean(matmul)

# 自定义metrics函数
# def accuracy(y_true, y_pred):
    

model = keras.Sequential()
model.add(LookupEmbedding(vocabulary_size,embedding_size))
model.add(layers.Dense(vocabulary_size,activation='softmax'))
#model.summary()
model.compile(optimizer='adam',loss=categorical_crossentropy,metrics=['accuracy'])
history = model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=10,validation_split=0.2,validation_data=None,validation_steps=20)

e = model.layers[0]
weights = e.get_weights()[0]

# 为embeddings编写相应的标签。
with open(os.path.join(FLAGS.output,'meta.tsv'), 'w') as f:
    for i in range(vocabulary_size):
        f.write(reverse_dictionary[i] + '\n')
    
with open(os.path.join(FLAGS.output,'vecs.tsv'), 'w') as f:
    for value in weights:
        f.write('\t'.join([str(x) for x in value]) + "\n")

