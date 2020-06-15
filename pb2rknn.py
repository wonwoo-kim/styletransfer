from rknn.api import RKNN

rknn = RKNN()

print('--> loading model')

ret = rknn.load_tensorflow(tf_pb='./output_graph.pb',
		inputs=['X_content'],
		outputs=['add_37'],
		input_size_list=[[20,256,256,3]])

if ret !=0:
	print('load failed!')
	exit(ret)


print('done')


