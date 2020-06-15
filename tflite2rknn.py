from rknn.api import RKNN

rknn = RKNN(verbose=True)

print('--> loading model')

ret = rknn.load_tflite(model='./ts.tflite')

if ret !=0:
	print('load failed!')
	exit(ret)

print('done')



