import numpy as np
import matplotlib.pyplot as plt

# slope and thre are deprecated
def dc_iv_preproc(
	vg, vd, ids, 
	scale, shift, 
	**kwarg
):
	if len(kwarg) > 0:
		print('[WARNING]: slope and threshold preprocessing is deprecated!')
	preproc_vg = (vg-shift) / scale['vg']
	preproc_vd = vd / scale['vd']
	preproc_id = ids / scale['id']
	return preproc_vg, preproc_vd, preproc_id

def get_restore_id_func(scale, *arg, **kwarg):
	if (len(arg) > 0 or len(kwarg) > 0):
		print('[WARNING]: slope and threshold preprocessing is deprecated!')
	def restore_id_func(ids, *arg):
		return ids * scale['id']
	def get_restore_id_grad_func(sig_grad, tanh_grad):
		return (sig_grad * scale['id'] / scale['vg'],
			tanh_grad * scale['id'] / scale['vd'])

	return restore_id_func, get_restore_id_grad_func



def compute_dc_meta(vg, vd, ids):
	vg_shift = np.median(vg)-0.0
	vg_scale = max(abs(np.max(vg)-vg_shift)/1.0, abs(np.min(vg)-vg_shift)/1.0)
	vd_scale = max(abs(np.max(vd))/1.0, abs(np.min(vd))/1.0)
	id_scale = max(abs(np.max(ids))/0.75, abs(np.min(ids))/0.75)

	scale = {'vg':vg_scale, 'vd':vd_scale, 'id':id_scale}

	return scale, vg_shift


def truncate(data_arrays, truncate_range, axis):
	# The data within the truncate_range will be removed, the rest will be returned.

    assert (truncate_range[0] <= np.max(data_arrays[axis]) and truncate_range[1] > np.min(data_arrays[axis])),\
		'Truncate range is out of data range, abort!'

    index = np.logical_not(
		np.logical_and(
			data_arrays[axis] > truncate_range[0],
			data_arrays[axis] <= truncate_range[1]
		)
	)

    return [e[index] for e in data_arrays]

#AC QV preproc
def ac_qv_preproc(preproc_voltages, gradient, scale, shift):
	preproc_voltages[:,0] = preproc_voltages[:,0]+shift
	v_scale = scale['v']
	for v in range(preproc_voltages.shape[1]):
		preproc_voltages[:, v] /= v_scale[v]
	preproc_gradient = gradient/scale['q']
	return preproc_voltages, preproc_gradient

def restore_voltages(scale, shift, voltages):
	v_scale = scale['v']
	for v in range(voltages.shape[1]):
		voltages[:, v] *= v_scale[v]
	voltages[:, 0] += shift
	return voltages

def get_restore_q_func(
        scale, shift
):
    def restore_integral_func(integrals):
        ori_integral = integrals*scale['q']
        for v in scale['v']:
        	ori_integral *= ori_integral*v
        return ori_integral
    def restore_gradient_func(gradient):
        ori_gradient =  gradient * scale['q']
        return ori_gradient
    return restore_integral_func, restore_gradient_func
    
def compute_ac_meta(voltage, gradient):
    vg = voltage[:, 0]
    vg_shift = np.median(vg)-0.0
    v_scale = [max(abs(np.max(vg)-vg_shift)/1.0, abs(np.min(vg)-vg_shift)/1.0)]
    for v in range(voltage.shape[1]):
    	if v != 0:
    		v_scale.append(max(abs(np.max(v))/1.0, abs(np.min(v))/1.0))
    q_scale = max(abs(np.max(gradient))/0.75, abs(np.min(gradient))/0.75)  
    scale = {'v':v_scale, 'q':q_scale}
    return scale, vg_shift


def plotgradient(vg, vd, gradient):
    #gradient = np.asarray(gradient)
    dqdvg = gradient[:, 0]
    dqdvd = gradient[:, 1]
    plt.plot(vg, dqdvg, 'r')
    plt.plot(vd, dqdvd, 'b')
    plt.show()

#Adjoint pinn
def adjoint_pinn_preproc(sig_input, tanh_input, sig_label, tanh_label, scale, shift):
	sig_input = (sig_input-shift)/scale['sig_input']
	tanh_input /= scale['tanh_input']
	sig_label /= scale['sig_label']
	tanh_label /= scale['tanh_label']
	return [sig_input, tanh_input, sig_label, tanh_label]

def restore_inputs(sig_input, tanh_input, scale, shift):
	sig_input = (sig_input*scale['sig_input']) + shift
	tanh_input *= scale['tanh_input']
	return sig_input, tanh_input

def compute_adjoint_pinn_meta(data_arrays):
	assert len(data_arrays == 4), 'Wrong number of input data'
	sig_input = data_arrays[0]
	tanh_input = data_arrays[1]
	sig_label = data_arrays[2]
	tanh_label = data_arrays[3]
	shift = np.median(data_arrays[0])-0.0
	sig_input_scale = max(abs(np.max(sig_input)-shift)/1.0, abs(np.min(sig_input)-shift)/1.0)
	tanh_input_scale = max(abs(np.max(tanh_input))/1.0, abs(np.min(tanh_input))/1.0)
  	sig_label_scale = max(abs(np.max(sig_label))/1.0, abs(np.min(sig_label))/1.0)
  	tanh_label_scale = max(abs(np.max(tanh_label))/1.0, abs(np.min(tanh_label))/1.0)
  	scale = {'sig_input':sig_input_scale, 'tanh_input':tanh_input_scale, 'sig_label':sig_label_scale, 'tanh_label':tanh_label_scale}
  	return scale, shift

def get_restore_adjoint_pinn_func(scale, shift):
	def restore_sig_output(sig_output):
		ori_sig_output = sig_output*scale['sig_label']
		return ori_sig_output
	def restore_tanh_output(tanh_output):
		ori_tanh_output = tanh_output*scale['tanh_label']
		return ori_tanh_output
	return restore_sig_output, restore_tanh_output

