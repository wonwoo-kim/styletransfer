import tensorflow as tf

meta_path = 'fns.ckpt.meta'
output_node_names = ['add_37']

with tf.compat.v1.Session() as sess:
	saver = tf.train.import_meta_graph(meta_path)

	saver.restore(sess, tf.train.latest_checkpoint('.'))

	frozen_graph_def = tf.graph_util.convert_variables_to_constants( sess, sess.graph_def, output_node_names)

	with open('output_graph.pb', 'wb') as f:
		f.write(frozen_graph_def.SerializeToString())
