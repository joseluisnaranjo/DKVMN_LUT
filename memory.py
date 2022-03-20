import numpy as np
import os
import tensorflow as tf
import operations


# This class defines Memory architecture in DKVMN  定义记忆矩阵   以key阵，20，50传入数据理解
class DKVMN_Memory():
    def __init__(self, memory_size, memory_state_dim, name):
        self.name = name
        print('%s initialized' % self.name)
        # Memory size : N	记忆矩阵的规模为N，20
        self.memory_size = memory_size
        # Memory state dim : D_V or D_K 200，50
        self.memory_state_dim = memory_state_dim
        '''
			Key matrix or Value matrix K矩阵或者V矩阵
			Key matrix is used for calculating correlation weight(attention weight) K阵计算相关权重
		'''

    # 相关权重的计算 32*50，20*50
    def cor_weight(self, embedded, key_matrix):
        '''
			embedded : [batch size, memory state dim(d_k)]
			Key_matrix : [memory size * memory state dim(d_k)]
			Correlation weight : w(i) = k * Key matrix(i)
			=> batch size * memory size
		'''
        # embedding_result : [batch size, memory size], each row contains each concept correlation weight for 1 question
        embedding_result = tf.matmul(embedded, tf.transpose(key_matrix))  # 做了格式转换transpose，行列转置。然后相乘
        correlation_weight = tf.nn.softmax(embedding_result)            #归一化到0到1之间
        print('Correlation weight shape : %s' % (correlation_weight.get_shape()))
        return correlation_weight

    # 计算读取内容 read content： 相关权重和答题情况的乘积
    # Getting read content 读过程，参数为32*20*200的value矩阵和32*20的w注意力权重阵
    def read(self, value_matrix, correlation_weight):
        '''
			Value matrix : [batch size ,memory size ,memory state dim] 三维矩阵
			Correlation weight : [batch size ,memory size], each element represents each concept embedding for 1 question
		'''
        # Reshaping
        # [batch size * memory size, memory state dim(d_v)]
        # reshape改变格式，value_matrix变为memory_state_dim200 维度的矩阵，行数自动计算  得出640*200矩阵
        vmtx_reshaped = tf.reshape(value_matrix, [-1, self.memory_state_dim])
        # [batch size * memory size, 1] 相关权重格式变换，行数自动计算,结果是640维度的列向量。
        cw_reshaped = tf.reshape(correlation_weight, [-1, 1])
        print('Transformed shape : %s, %s' % (vmtx_reshaped.get_shape(), cw_reshaped.get_shape()))
        # Read content, will be [batch size * memory size, memory state dim] and reshape it to [batch size, memory size, memory state dim]
        rc = tf.multiply(vmtx_reshaped, cw_reshaped)    #得到640*200张量
        # print(rc.get_shape())
        read_content = tf.reshape(rc, [-1, self.memory_size, self.memory_state_dim])  # read_content的格式转换
        # Summation through memory size axis, make it [batch size, memory state dim(d_v)]
        read_content = tf.compat.v1.reduce_sum(read_content, axis=1, keep_dims=False)  # reduce_sum（）降维求和，纵向求和压缩列，20个求和
        print('Read content shape : %s' % (read_content.get_shape()))   #32*200阵
        return read_content

    # 写过程
    def write(self, value_matrix, correlation_weight, qa_embedded, reuse=False):
        '''
			Value matrix : [batch size, memory size, memory state dim(d_k)]
			Correlation weight : [batch size, memory size]
			qa_embedded : (q, r) pair embedded, [batch size, memory state dim(d_v)]
		'''
        erase_vector = operations.linear(qa_embedded, self.memory_state_dim, name=self.name + '/Erase_Vector',
                                         reuse=reuse)
        # [batch size, memory state dim(d_v)]
        erase_signal = tf.sigmoid(erase_vector)
        add_vector = operations.linear(qa_embedded, self.memory_state_dim, name=self.name + '/Add_Vector', reuse=reuse)
        # [batch size, memory state dim(d_v)]
        add_signal = tf.tanh(add_vector)

        # Add vector after erase
        # [batch size, 1, memory state dim(d_v)]
        erase_reshaped = tf.reshape(erase_signal, [-1, 1, self.memory_state_dim])
        # [batch size, memory size, 1]
        cw_reshaped = tf.reshape(correlation_weight, [-1, self.memory_size, 1])
        # w_t(i) * e_t
        erase_mul = tf.multiply(erase_reshaped, cw_reshaped)
        # Elementwise multiply between [batch size, memory size, memory state dim(d_v)]
        erase = value_matrix * (1 - erase_mul)
        # [batch size, 1, memory state dim(d_v)]
        add_reshaped = tf.reshape(add_signal, [-1, 1, self.memory_state_dim])
        add_mul = tf.multiply(add_reshaped, cw_reshaped)

        new_memory = erase + add_mul
        # [batch size, memory size, memory value staet dim]
        print('Memory shape : %s' % (new_memory.get_shape()))
        return new_memory


# This class construct key matrix and value matrix  key阵和value阵的建立
class DKVMN():
    #20，50，200，20*50，32*20*200
    def __init__(self, memory_size, memory_key_state_dim, memory_value_state_dim, init_memory_key, init_memory_value,
                 name='DKVMN'):
        print('Initializing memory..')
        self.name = name
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim

        self.key = DKVMN_Memory(self.memory_size, self.memory_key_state_dim, name=self.name + '_key_matrix')
        self.value = DKVMN_Memory(self.memory_size, self.memory_value_state_dim, name=self.name + '_value_matrix')

        self.memory_key = init_memory_key
        self.memory_value = init_memory_value

    def attention(self, q_embedded):        #model memory对象调用，传递参数q，32*50
        correlation_weight = self.key.cor_weight(embedded=q_embedded, key_matrix=self.memory_key)
        return correlation_weight

    def read(self, c_weight):               #同上个函数用对象调用，参数为注意力权重矩阵，32*20
        read_content = self.value.read(value_matrix=self.memory_value, correlation_weight=c_weight)
        return read_content

    def write(self, c_weight, qa_embedded, reuse):
        self.memory_value = self.value.write(value_matrix=self.memory_value, correlation_weight=c_weight,
                                             qa_embedded=qa_embedded, reuse=reuse)
        return self.memory_value
