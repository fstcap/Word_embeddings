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

'''
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print("\033[0;33m",batch[i], reverse_dictionary[batch[i]],"\033[0m", '->',labels[i,0], reverse_dictionary[labels[i,0]])
'''


# Step 4: 创建和训练skip-gram model
batch_size = 128
embedding_size = 128 # 嵌入向量的维数。
skip_window = 1 # 左右要考虑多少个单词。
num_skips = 2 # 重复使用输入生成标签的次数。

class InputEmbedding(object):
    def __init__(self):
        self.kernels = tf.Variable(
            tf.random.uniform(
                shape=[vocabulary_size,embedding_size],
                minval=-1.0,
                maxval=1.0), name='input_kernels')
    def __call__(self, x):
        return tf.nn.embedding_lookup(params=self.kernels, ids=x)

class LabelEmbedding(object):
    def __init__(self):
        self.kernels = tf.Variable(
            tf.random.truncated_normal(
                shape=[vocabulary_size,embedding_size],
                stddev=1.0/math.sqrt(embedding_size)), name='label_kernels')
        self.biases = tf.Variable(
            tf.zeros(shape=[vocabulary_size]), name='label_biases')
    def __call__(self, x):
        embeddings = tf.nn.embedding_lookup(params=self.kernels, ids=x)
        biases = tf.nn.embedding_lookup(params=self.biases, ids=x)
        return embeddings, biases
        

def compute_sampled_logits(y_true, y_pred, num_true=1, num_sampled=64):
     
    y_true = tf.dtypes.cast(y_true,dtype=tf.int64)
    y_true_flat = tf.reshape(y_true, [-1])
    
    sampled_values = tf.random.log_uniform_candidate_sampler(
        true_classes=y_true,
        num_true=num_true,
        num_sampled=num_sampled,
        unique=True,
        range_max=vocabulary_size,
        seed=None)
    sampled, true_expected_count, sampled_expected_count = (tf.stop_gradient(s) for s in sampled_values)
    sampled = tf.dtypes.cast(sampled,dtype=tf.int64)

    all_ids = tf.concat([y_true_flat, sampled], 0)
    all_w, all_b = label_model(all_ids)
    
    true_w_shape = tf.stack([tf.shape(y_true_flat)[0], -1])
    true_w = tf.slice(input_=all_w, begin=[0,0], size=true_w_shape)
    
    sampled_w_shape = tf.stack([tf.shape(y_true_flat)[0], 0])
    sampled_w = tf.slice(input_=all_w, begin=sampled_w_shape, size=[-1, -1])
    sampled_logits = tf.linalg.matmul(y_pred, sampled_w, transpose_b=True)
    
    true_b = tf.slice(input_=all_b, begin=[0], size=tf.shape(y_true_flat))
    sampled_b = tf.slice(input_=all_b, begin=tf.shape(y_true_flat), size=[-1])

    dim = tf.shape(true_w)[1:2]
    new_true_w_shape = tf.concat([[-1, num_true], dim], 0)
    new_inputs = tf.expand_dims(y_pred, axis=1)
    row_wise_dots = tf.math.multiply(new_inputs, tf.reshape(true_w, new_true_w_shape))

    dots_as_matrix = tf.reshape(row_wise_dots, shape=tf.concat([[-1],dim],0))
    true_logits = tf.reshape(tf.math.reduce_sum(input_tensor=dots_as_matrix, axis=1), shape=[-1, num_true])
    true_b = tf.reshape(true_b, shape=[-1, num_true])
    true_logits += true_b
    sampled_logits += sampled_b
    
    true_logits -= tf.math.log(true_expected_count)
    sampled_logits -= tf.math.log(sampled_expected_count)
    out_logits = tf.concat([true_logits, sampled_logits], 1)

    out_labels = tf.concat([
        tf.ones_like(true_logits) / num_true,
        tf.zeros_like(sampled_logits)
    ], 1)

    return out_logits, out_labels
    
input_model = InputEmbedding()
label_model = LabelEmbedding()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def step_train(inputs,targets):
    with tf.GradientTape() as tape:
        y_pred = input_model(inputs) 
        logits, labels = compute_sampled_logits(targets,y_pred)
        sampled_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.math.reduce_sum(sampled_losses, axis=1)
    model_variables = [input_model.kernels, label_model.kernels, label_model.biases]
    grads = tape.gradient(loss, model_variables)
    optimizer.apply_gradients(zip(grads, model_variables))
    return tf.math.reduce_mean(loss)

train_loss_results, train_accuracy_results = [], []
test_loss_results, test_accuracy_results = [], []
num_epochs = 100001

average_loss = 0
for step in range(num_epochs):
    x, y = generate_batch(batch_size,num_skips,skip_window)
    y = tf.dtypes.cast(y,dtype=tf.float32)
    loss = step_train(x, y)  
    average_loss += loss
    if step % 100 == 0 and step > 0:
        average_loss /= 100
        train_loss_results.append(average_loss)
        print(f"\033[1;35mStep:\033[0m{step}",f"\033[1;36mAverage_Loss:\033[0m{average_loss}")
        average_loss = 0

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

weights = input_model.kernels
norm = tf.math.sqrt(tf.math.reduce_sum(input_tensor=tf.math.square(weights), axis=1, keepdims=True))
normalized_weights = weights/norm
print(f"\033[1;35mnormalized_weights shape:\033[0m{normalized_weights.shape}")

# 为embeddings编写相应的标签。
with open(os.path.join(FLAGS.output,'meta.tsv'), 'w') as f:
    for i in range(vocabulary_size):
        f.write(reverse_dictionary[i] + '\n')
    
with open(os.path.join(FLAGS.output,'vecs.tsv'), 'w') as f:
    for value in normalized_weights.numpy():
        f.write('\t'.join([str(x) for x in value]) + "\n")

end_time = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
print(f"\033[1;35mStart_time:\033[0m{start_time}\n\033[1;35mEnd_time:\033[0m{end_time}")
