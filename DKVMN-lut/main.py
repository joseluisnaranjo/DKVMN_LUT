import tensorflow._api.v2.compat.v1 as tf
import numpy as np
from model import Model
import os, time, argparse	##argpse模块为了向命令行传递参数
from data_loader import *
tf.compat.v1.disable_eager_execution()

def main():
	parser = argparse.ArgumentParser()							##首先创建ArgumentParser对象，作为一个解析器
	parser.add_argument('--num_epochs', type=int, default=10)		##对象调用添加属性方法，添加参数信息 num_epochs,整形，默认300
	parser.add_argument('--train', type=str2bool, default='t')		##自定义的str2bool类型，传入参数't'
	parser.add_argument('--init_from', type=str2bool, default='t')
	parser.add_argument('--show', type=str2bool, default='f')
	parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')		##log_dir和data_dir的值设定好了
	parser.add_argument('--log_dir', type=str, default='logs')
	parser.add_argument('--data_dir', type=str, default='data')
	parser.add_argument('--anneal_interval', type=int, default=20)
	parser.add_argument('--maxgradnorm', type=float, default=50.0)
	parser.add_argument('--momentum', type=float, default=0.9)
	parser.add_argument('--initial_lr', type=float, default=0.05)
	# synthetic / assist2009_updated / assist2015 / STATIC
	dataset = 'assist2009_updated'		#数据集的选择，根据不同的数据集再定义相关的数据格式

	if dataset == 'assist2009_updated':
		parser.add_argument('--batch_size', type=int, default=32)
		parser.add_argument('--memory_size', type=int, default=20)
		parser.add_argument('--memory_key_state_dim', type=int, default=50)
		parser.add_argument('--memory_value_state_dim', type=int, default=200)
		parser.add_argument('--final_fc_dim', type=int, default=50)
		parser.add_argument('--n_questions', type=int, default=110)
		parser.add_argument('--seq_len', type=int, default=200)

	elif dataset == 'synthetic':
		parser.add_argument('--batch_size', type=int, default=32)
		parser.add_argument('--memory_size', type=int, default=5)
		parser.add_argument('--memory_key_state_dim', type=int, default=10)
		parser.add_argument('--memory_value_state_dim', type=int, default=10)
		parser.add_argument('--final_fc_dim', type=int, default=50)
		parser.add_argument('--n_questions', type=int, default=50)
		parser.add_argument('--seq_len', type=int, default=50)

	elif dataset == 'assist2015':
		parser.add_argument('--batch_size', type=int, default=50)
		parser.add_argument('--memory_size', type=int, default=20)
		parser.add_argument('--memory_key_state_dim', type=int, default=50)
		parser.add_argument('--memory_value_state_dim', type=int, default=100)
		parser.add_argument('--final_fc_dim', type=int, default=50)
		parser.add_argument('--n_questions', type=int, default=100)
		parser.add_argument('--seq_len', type=int, default=200)

	elif dataset == 'STATICS':
		parser.add_argument('--batch_size', type=int, default=10)
		parser.add_argument('--memory_size', type=int, default=50)
		parser.add_argument('--memory_key_state_dim', type=int, default=50)
		parser.add_argument('--memory_value_state_dim', type=int, default=100)
		parser.add_argument('--final_fc_dim', type=int, default=50)
		parser.add_argument('--n_questions', type=int, default=1223)
		parser.add_argument('--seq_len', type=int, default=200)
	##参数解析parse_args()
	args = parser.parse_args()
	args.dataset = dataset		#加上dataset属性

	print(args)
	if not os.path.exists(args.checkpoint_dir):
		os.mkdir(args.checkpoint_dir)
	if not os.path.exists(args.log_dir):
		os.mkdir(args.log_dir)
	if not os.path.exists(args.data_dir):
		os.mkdir(args.data_dir)
		raise Exception('Need data set')

	run_config = tf.compat.v1.ConfigProto()
	run_config.gpu_options.allow_growth = True	#GPU使用权限

	#数据加载，已经定义好了默认的值，路径
	data = DATA_LOADER(args.n_questions, args.seq_len, ',')
	data_directory = os.path.join(args.data_dir, args.dataset)

	with tf.compat.v1.Session(config=run_config) as sess:
		dkvmn = Model(args, sess, name='DKVMN')
		if args.train:
			train_data_path = os.path.join(data_directory, args.dataset + '_train1.csv')
			valid_data_path = os.path.join(data_directory, args.dataset + '_valid1.csv')

			train_q_data, train_qa_data = data.load_data(train_data_path)
			print('Train data loaded')
			valid_q_data, valid_qa_data = data.load_data(valid_data_path)
			print('Valid data loaded')
			print('Shape of train data : %s, valid data : %s' % (train_q_data.shape, valid_q_data.shape))
			print('Start training')
			dkvmn.train(train_q_data, train_qa_data, valid_q_data, valid_qa_data)		#模型对象调用训练函数，开始训练
			#print('Best epoch %d' % (best_epoch))
		else:		#测试用
			test_data_path = os.path.join(data_directory, args.dataset + '_test.csv')
			test_q_data, test_qa_data = data.load_data(test_data_path)
			print('Test data loaded')
			dkvmn.test(test_q_data, test_qa_data)



def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):		#lower函数转换字符串字母为小写
		return True
	elif v.lower() in ('no', 'false', 'n', 'f', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Not expected boolean type')

if __name__ == "__main__":		#魔法函数__name__,用于模块执行导入时候分割，设置入口
	main()



