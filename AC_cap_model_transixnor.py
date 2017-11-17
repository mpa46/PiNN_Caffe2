from ac_qv_api import ACQVModel, plot_iv, predict_qs
import pinn.parser as parser
import pinn.preproc as preproc
import pinn.exporter as exporter
import pinn.deembed as deembed
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

vds = np.zeros(41*41*67, dtype=np.float32)
vbg = np.zeros(41*41*67, dtype=np.float32)
vtg = np.zeros(41*41*67, dtype=np.float32)

#Vds array
for x in range(41):
	for y in range(41*67):
		vds[x*41*67+y] = x*(0.4/40.0)

#Vbg array
for x in range(41):
	for y in range(41):
		for z in range(67):
			vbg[x*41*67+y*67+z] = y*(0.4/40.0)

#Vtg array
for x in range(41*41):
	for y in range(67):
		vtg[x*67+y] = y*(0.4/66.0) 



voltage = np.concatenate(
	(np.expand_dims(vds, axis=1),
	 np.expand_dims(vbg, axis=1),
	 np.expand_dims(vtg, axis=1)), 
	axis=1
)



data_array_1 = np.load ('./TransiXNOR_data/vds_vbg_vtg_Cbd_D2_AC.npy')
data_array_2 = np.load ('./TransiXNOR_data/vds_vbg_vtg_Cbg_D2_AC.npy')
data_array_3 = np.load ('./TransiXNOR_data/vds_vbg_vtg_Cbs_D2_AC.npy')

capas = np.concatenate(
	(np.expand_dims(data_array_1.flatten(), axis=1),
	 np.expand_dims(data_array_2.flatten(), axis=1),
	 np.expand_dims(data_array_3.flatten(), axis=1)), 
	axis=1
)

scale, vg_shift = preproc.compute_ac_meta(voltage, capas)


preproc_param = {
	'scale': scale,
	'vg_shift': vg_shift,
}

ac_model = ACQVModel('ac_model', input_dim=3, output_dim=1)
ac_model.add_data('train', [voltage, capas], preproc_param)
ac_model.build_nets(
	[16, 16], 
	batch_size = 1024,
	optim_method = 'AdaGrad',
	optim_param = {'alpha':0.1, 'epsilon':1e-4},
)
ac_model.train_with_eval(
	num_epoch = 10, 
	report_interval = 0,
)
#ac_model.draw_nets() 
#ac_model.plot_loss_trend()

pred_qs, ori_qs, pred_grad, ori_grad = ac_model.predict_qs(
	voltage)

# Plot predicted q
plot_iv(
	voltage[:, 0], voltage[:, 1], ori_qs
)

plot_iv(
	voltage[:, 0], voltage[:, 1], ori_grad[:, 0],
	voltage[:, 0], voltage[:, 1], capas[:, 0],
	styles = ['vg_major_linear', 'vd_major_linear']
)

plot_iv(
	voltage[:, 0], voltage[:, 1], ori_grad[:, 2],
	voltage[:, 0], voltage[:, 1], capas[:, 2],
	styles = ['vg_major_linear', 'vd_major_linear']
)

plot_iv(
	voltage[:, 0], voltage[:, 1], ori_grad[:, 2],
	voltage[:, 0], voltage[:, 1], capas[:, 2],
	styles = ['vg_major_linear', 'vd_major_linear']
)
