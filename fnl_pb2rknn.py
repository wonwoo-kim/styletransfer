from rknn.api import RKNN

rknn = RKNN()

print('--> loading model')




ret = rknn.load_tensorflow(tf_pb='./trained_graph.pb',
		inputs=['x_p'],
		outputs=['hypothesis'],
		input_size_list=[[1,1]])

if ret !=0:
	print('load failed!')
	exit(ret)


print('rknn converting done')

print('--> building model')

ret = rknn.build(do_quantization=False)

if ret !=0:
	print('build failed!')
	exit(ret)

print('build model done')



print('--> export rknn model')

ret = rknn.export_rknn('./sample_trained.rknn')

if ret !=0:
	print("export failed")
	exit(ret)

print('export done')






