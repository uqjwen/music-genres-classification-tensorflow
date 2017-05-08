from data_loader import Data_Loader 
from model import Model 
import tensorflow as tf 
from config import numEpoches
from config import batchSize
from config import checkEvery
import sys


def main():
	data_helper = Data_Loader()
	genres = len(data_helper.genres)

	model = Model(genres)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		saver = tf.train.Saver(tf.all_variables(), max_to_keep=20)
		checkpoint_dir = './checkpoint/'
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print (" [*] Loading parameters success...")
		else:
			print (" [!] Loading parameters failed!!!")



		for e in range(numEpoches):
			data_helper.set_pointer()
			total_batch = int(len(data_helper.train_x)/batchSize)
			for b in range(total_batch):
				x,y = data_helper.next_batch()
				feed = {model.x: x, model.y:y}
				_, loss, acc = sess.run([model.train_op, model.loss, model.acc], feed_dict = feed)
				sys.stdout.write("\r {} epoch,{} batch, train_loss:{}, train_acc:{}".format(e,b,loss,acc))
				sys.stdout.flush()

				if( ((e*total_batch+b)!=0 and(e*total_batch+b)%checkEvery==0) or (e == numEpoches-1 and b == total_batch-1)):
					x,y = data_helper.test_data()
					feed = {model.x:x, model.y:y}
					loss,acc = sess.run([model.loss, model.acc], feed_dict = feed)
					print("\n val_loss:{}, val_acc:{}".format(loss, acc))
					saver.save(sess, checkpoint_dir+'model.ckpt', global_step = e*total_batch+b)


if __name__ == '__main__':
	main()