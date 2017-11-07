from pinn_api import DeviceModel, plot_iv, predict_ids_grads
import pinn.parser as parser
import pinn.preproc as preproc
import pinn.exporter as exporter
import pinn.deembed as deembed
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from caffe2.python import optimizer
import math
from pinn.adjoint_pinn_lib import (
	build_adjoint_pinn, init_model_with_schemas, TrainTarget)

def f(x,y):
    return math.tanh(5*y)*(1/(1+math.exp(0-5*x)))
x = y = np.arange(-1, 1, 0.2)
X, Y = np.meshgrid(x, y)
zs = np.array([f(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
# np.gradient reverse the order
gy, gx = np.gradient(Z, 0.2, 0.2)

data_arrays = [np.ravel(X), np.ravel(Y), np.ravel(gx), np.ravel(gy)]
data_arrays[0] = np.array(np.expand_dims(data_arrays[0], axis = 1), dtype=np.float32)
data_arrays[1] = np.array(np.expand_dims(data_arrays[1], axis = 1), dtype=np.float32)
data_arrays[2] = np.array(np.expand_dims(data_arrays[2], axis = 1), dtype=np.float32)
data_arrays[3] = np.array(np.expand_dims(data_arrays[3], axis = 1), dtype=np.float32)

scale, vg_shift = preproc.compute_dc_meta (data_arrays[0], data_arrays[1], data_arrays[2])
preproc_param = {
	'scale' : scale,
	'vg_shift' : vg_shift,
}

device_model = DeviceModel('pinn_model', 1, 1, 1, train_target=TrainTarget.ADJOINT)
device_model.add_data('train', data_arrays, preproc_param)
device_model.build_nets (
	hidden_sig_dims=[10, 1], 
	hidden_tanh_dims=[10, 1],
	train_batch_size=100,
	weight_optim_param={'alpha':0.01, 'epsilon':1e-4},
	bias_optim_param={'alpha':0.01, 'epsilon':1e-4},
	max_loss_scale=1.1
	)
device_model.train_with_eval (num_epoch=10000)
_ids, ids = device_model.predict_ids(data_arrays[0], data_arrays[1],)

plot_iv (data_arrays[0], data_arrays[1], ids, 
	vg_comp=data_arrays[0], vd_comp=data_arrays[1], ids_comp=data_arrays[2])

plot_iv (data_arrays[0], data_arrays[1], ids,
	vg_comp=data_arrays[0], vd_comp=data_arrays[1], ids_comp=data_arrays[3])


