# 数据加载规格化
import numpy as np
import tensorflow as tf
from six.moves import xrange


# six.moves作用，兼容python2和3版本 https://www.cnblogs.com/charlesblc/p/8027289.html
# xrange的用法类似于range，不过是用来生成一个生成器，而不是数组。xrange(开始，停止（不到停止点），步长)
# https://www.runoob.com/python/python-func-xrange.html
class DATA_LOADER():
    # 初始化，使用了魔法函数，self为惯例第一个参数名称。https://www.runoob.com/python3/python3-class.html
    def __init__(self, n_questions, seqlen, seperate_char):
        # assist2009 : seq_len(200), n_questions(110)
        # Each value is seperated by seperate_char
        self.seperate_char = seperate_char  # 分隔符号，这里为逗号
        self.n_questions = n_questions  # 回答问题的数目，assist2009_updated数据集里是110
        self.seq_len = seqlen  # 序列长度，assist2009_updated数据集里是220

    '''
	Data format as followed
	1) Number of exercies
	2) Exercise tag
	3) Answers
	数据格式：习题数目、习题标签、习题的回答答案
	'''

    # path : data location  路径：数据位置
    # 定义加载数据的函数，加载数据的路径
    def load_data(self, path):  # main函数里data对象调用，参数传递的是相应数据集下train，valid,test的路径
        # open内置函数打开数据集，open具体用法 https://www.runoob.com/python/python-func-open.html
        f_data = open(path, 'r')
        # Question/Answer container 问题和答案容器，定义两个列表类型。
        q_data = list()
        qa_data = list()
        # Read data	读取数据
        # enumerate() 将列表元组或者字符串加上索引，并且可以指定索引开始，默认下标从零开始。csv文件里按照每行作为一个索引下表进行编号。
        # linedid这里即表示习题的索引号。
        for lineid, line in enumerate(f_data):  # 这里lined为索引号，line表示f_data中的内容。
            # strip去掉所有数据中首尾空格字符，strip()默认去开头和结尾的空格字符串，可以指定开头结尾的字符串。
            line = line.strip()
            # Exercise tag line 习题标签处理
            if lineid % 3 == 1:  # 默认从零开始，按三三排序最后得出所有中间的是习题标签。
                # split by ',', returns tag list
                print('Excercies tag'),
                # split是用来切片字符串的函数，返回列表，有两个参数，第一个表示分割符号，默认是所有空字符，包括换行之类的。第二个表示分割大小，默认全分割。
                # 习题标签按逗号分隔后放在一个列表中，q_tag_list列表。
                q_tag_list = line.split(self.seperate_char)

            # Answer 回答操作
            elif lineid % 3 == 2:  # 三三行排列最后一行回答的操作
                print(', Answers')
                answer_list = line.split(self.seperate_char)  # 将回答按逗号为分隔符存入回答列表中

                # Divide case by seq_len 按问题长度划分。按照问题长度划分，看问题标签的数目。
                if len(q_tag_list) > self.seq_len:  # 每行习题标签数目大于规定长度(assist2009_updated里设定是200)时，
                    n_split = len(q_tag_list) // self.seq_len  # 用整数除法，得出能满足最大固定长度的个数，最后应该会有小于200剩余的
                    if len(q_tag_list) % self.seq_len:  # 能整除的话，表示刚好足够，否则不足部分相当于加入上一个200的seq_len
                        n_split += 1
                else:  # 标签小于问题长度200，用一个seq_len即可满足
                    n_split = 1
                print('Number of split : %d' % n_split)  # 输出分裂的个数，需要seq_len的个数

                # Contain as many as seq_len, then contain remainder
                for k in range(n_split):        #range不包含右边界
                    q_container = list()  # 定义了两个空列表容器
                    qa_container = list()
                    # Less than 'seq_len' element remained
                    if k == n_split - 1:
                        end_index = len(answer_list)  # 结束指标的选择，就是说，在最后一个分裂内的结束标准不同，应对最后一个不满的情况
                    else:
                        end_index = (k + 1) * self.seq_len
                    for i in range(k * self.seq_len, end_index):    #按照每一个seq_len长度的内容输出问题标签，回答，和qa值
                        # answers in {0,1}，计算qa值，答对值得出qa更大，
                        qa_values = int(q_tag_list[i]) + int(answer_list[i]) * self.n_questions
                        q_container.append(int(q_tag_list[i]))
                        qa_container.append(qa_values) # esta valor consiste en el numero de ejercicio mas (num preguntas) o mas zero dependiendo de si la restuesta al el ejercicio fue correcta o no
                        print('Question tag : %s, Answer : %s, QA : %s' % (q_tag_list[i], answer_list[i], qa_values))
                    # List of list(seq_len, seq_len, seq_len, less than seq_len, seq_len, seq_len...
                    #两个列表，内容元素也为列表，q_data的元素为q_container,q_container为存储每一位学生答题标签的列表
                    q_data.append(q_container)      #每位学生的q_data中有n_split个q_container
                    qa_data.append(qa_container)
        f_data.close()
        # print(len(q_data))试试q_data多长，在assist2009_updated数据集情况下为740，即所有用到seq_len有740，有不满200，有超过200的学生
        # Convert it to numpy array	转换为numpy矩阵
        q_data_array = np.zeros((len(q_data), self.seq_len))    #q_data_arrayes una matriz con todos las preguntas en filas de 200 y completado con zeros en por cada ejercicio si no fuere multiplo de 200 que es la mayoria de casos
        for i in range(len(q_data)):                            #该数据集情况下从0到739循环，
            data = q_data[i]                                    #data中间变量
            # if q_data[i] less than seq_len, remainder would be 0
            q_data_array[i, :len(data)] = data                  #输出数值长度为data长度，不足部分为0，本数据集中<200的补上0
                                                                #740*200的矩阵
        #print(q_data_array) 看看矩阵
        qa_data_array = np.zeros((len(qa_data), self.seq_len))
        for i in range(len(qa_data)):
            data = qa_data[i]
            # if qa_data[i] less than seq_len, remainder would be 0
            qa_data_array[i, :len(data)] = data

        return q_data_array, qa_data_array
