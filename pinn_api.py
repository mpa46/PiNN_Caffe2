import caffe2_paths
import os
import pickle
from caffe2.python import (
	workspace, layer_model_helper, schema, optimizer, net_drawer
)
import caffe2.python.layer_model_instantiator as instantiator
from caffe2.python.layers.tags import Tags
import numpy as np
from pinn.adjoint_pinn_lib import (
	build_adjoint_pinn, init_model_with_schemas, TrainTarget)
import pinn.data_reader as data_reader
import pinn.preproc as preproc
import pinn.parser as parser
import pinn.visualizer as visualizer
import pinn.exporter as exporter
# import logging
import matplotlib.pyplot as plt

class DeviceModel(object):
	def __init__(
		self, 
		model_name,
		sig_input_dim=1,
		tanh_input_dim=1,
		output_dim=1,
		train_target=TrainTarget.ORIGIN
	):	
		self.model_name = model_name
		self.sig_input_dim = sig_input_dim
		self.tanh_input_dim = tanh_input_dim
		self.model = init_model_with_schemas(
			model_name, sig_input_dim, tanh_input_dim, output_dim, 
			train_target=train_target
		)
		self.input_data_store = {}
		self.preproc_param = {}
		self.net_store = {}
		self.reports = {
			'epoch':[],
			'train_loss':[], 'eval_loss':[],
			'train_l1_metric':[], 'eval_l1_metric':[],
			'train_scaled_l1_metric':[], 'eval_scaled_l1_metric':[]
		}
		self.train_target = train_target
		if train_target == TrainTarget.ORIGIN:
			self.adjoint_tag = Tags.PREDICTION_ONLY
		if train_target == TrainTarget.ADJOINT:
			self.adjoint_tag = Tags.EXCLUDE_FROM_PREDICTION

	def add_data(
		self,
		data_tag,
		data_arrays, 
		preproc_param,
		override=True,
	):
		'''
		data_arrays are in the order of 
			1) for train origin: sig_input, tanh_input, and label
			2) for train adjoint: sig_input, tanh_input, sig_adjoint_label and 
			   tanh_adjoint_label
		'''
		assert (
			(len(data_arrays) == 3 and self.train_target == TrainTarget.ORIGIN) or
			(len(data_arrays) == 4 and self.train_target == TrainTarget.ADJOINT)
			), 'Incorrect number of input data'

		# number of examples and same length assertion
		num_example = len(data_arrays[0])
		for data in data_arrays[1:]:
			assert len(data) == num_example, 'Mismatch dimensions'
	
		self.preproc_param = preproc_param

		self.pickle_file_name = self.model_name + '_preproc_param' + '.p'
		db_name = self.model_name + '_' + data_tag + '.minidb'

		if os.path.isfile(db_name):
			if override:
				print("XXX Delete the old database...")
				os.remove(db_name)
				os.remove(self.pickle_file_name)
			else:
				raise Exception('Encounter database with the same name. ' +
					'Choose the other model name or set override to True.')
		print("+++ Create a new database...")	
		pickle.dump(
			self.preproc_param, 
			open(self.pickle_file_name, 'wb')
		)

		if self.train_target == TrainTarget.ORIGIN:
			preproc_data_arrays = preproc.dc_iv_preproc(
				data_arrays[0], data_arrays[1], data_arrays[2], 
				self.preproc_param['scale'], 
				self.preproc_param['vg_shift']
			)
		if self.train_target == TrainTarget.ADJOINT:
			adjoint_input = np.ones((data_arrays[0].shape[0], 1))
			preproc_data_arrays = [data_arrays[0],
									data_arrays[1],
									adjoint_input,
									data_arrays[2],
									data_arrays[3]]

		self.preproc_data_arrays=preproc_data_arrays
		# Only expand the dim if the number of dimension is 1
		preproc_data_arrays = [np.expand_dims(
			x, axis=1) if x.ndim == 1 else x for x in preproc_data_arrays]
		
		# Write to database
		data_reader.write_db(
			'minidb', 
			db_name, 
			preproc_data_arrays,
		)
		self.input_data_store[data_tag] = [db_name, num_example]

	def build_nets(
		self,
		hidden_sig_dims, 
		hidden_tanh_dims,
		train_batch_size=1,
		eval_batch_size=1,
		weight_optim_method='AdaGrad',
		weight_optim_param={'alpha':0.01, 'epsilon':1e-4},
		bias_optim_method='AdaGrad',
		bias_optim_param={'alpha':0.01, 'epsilon':1e-4},
		loss_function='scaled_l1', 
		max_loss_scale=1.,
	):
		assert len(self.input_data_store) > 0, 'Input data store is empty.'
		assert 'train' in self.input_data_store, 'Missing training data.'
		self.batch_size = train_batch_size
		# Build the date reader net for train net
		if self.train_target == TrainTarget.ORIGIN:
			input_data_train = data_reader.build_input_reader(
				self.model, 
				self.input_data_store['train'][0], 
				'minidb', 
				['sig_input', 'tanh_input', 'label'], 
				batch_size=train_batch_size,
				data_type='train',
			)
			if 'eval' in self.input_data_store:
				# Build the data reader net for eval net
				input_data_eval = data_reader.build_input_reader(
					self.model, 
					self.input_data_store['eval'][0], 
					'minidb', 
					['eval_sig_input', 'eval_tanh_input', 'eval_label'], 
					batch_size=eval_batch_size,
					data_type='eval',
				)
			# for training origin, use origin_loss_record
			self.model.trainer_extra_schema.origin_loss_record.label.set_value(
				input_data_train[2].get(), unsafe=True)

		if self.train_target == TrainTarget.ADJOINT:
			input_data_train = data_reader.build_input_reader(
				self.model,
				self.input_data_store['train'][0],
				'minidb',
				['sig_input', 'tanh_input', 'adjoint_input', 'sig_adjoint_label', 'tanh_adjoint_label'],
				batch_size=train_batch_size,
				data_type='train',
			)
			if 'eval' in self.input_data_store:
				input_data_eval = data_reader.build_input_reader(
					self.model,
					self.input_data_store['eval'][0],
					'minidb',
					['eval_sig_input', 'eval_tanh_input', 'eval_adjoint_label', 'eval_sig_adjoint_label', 'eval_tanh_adjoint_label'],
					batch_size=eval_batch_size,
					data_type='eval',
				)
			self.model.input_feature_schema.adjoint_input.set_value(
				input_data_train[2].get(), unsafe=True)
			self.model.trainer_extra_schema.sig_loss_record.label.set_value(
				input_data_train[3].get(), unsafe=True)
			self.model.trainer_extra_schema.tanh_loss_record.label.set_value(
				input_data_train[4].get(), unsafe=True)


		# Build the computational nets
		# Create train net
		self.model.input_feature_schema.sig_input.set_value(
			input_data_train[0].get(), unsafe=True)
		self.model.input_feature_schema.tanh_input.set_value(
			input_data_train[1].get(), unsafe=True)

		(self.pred, self.sig_adjoint_pred, 
			self.tanh_adjoint_pred, self.loss) = build_adjoint_pinn(
			self.model,
			sig_input_dim=self.sig_input_dim,
			tanh_input_dim=self.tanh_input_dim,
			sig_net_dim=hidden_sig_dims,
			tanh_net_dim=hidden_tanh_dims,
			weight_optim=_build_optimizer(weight_optim_method, weight_optim_param),
			bias_optim=_build_optimizer(bias_optim_method, bias_optim_param),
			adjoint_tag=self.adjoint_tag,
			train_target=self.train_target,
			loss_function=loss_function,
			max_loss_scale=max_loss_scale,
		)

		train_init_net, train_net = instantiator.generate_training_nets(self.model)
		workspace.RunNetOnce(train_init_net)
		workspace.CreateNet(train_net)
		self.net_store['train_net'] = train_net

		pred_net = instantiator.generate_predict_net(self.model)
		workspace.CreateNet(pred_net)
		self.net_store['pred_net'] = pred_net

		if 'eval' in self.input_data_store:
			# Create eval net
			self.model.input_feature_schema.sig_input.set_value(
				input_data_eval[0].get(), unsafe=True)
			self.model.input_feature_schema.tanh_input.set_value(
				input_data_eval[1].get(), unsafe=True)

			if self.train_target == TrainTarget.ORIGIN: 
				self.model.trainer_extra_schema.origin_loss_record.label.set_value(
					input_data_eval[2].get(), unsafe=True)

			if self.train_target == TrainTarget.ADJOINT:
				self.model.input_feature_schema.adjoint_input.set_value(
					input_data_eval[2].get(), unsafe=True)
				self.model.trainer_extra_schema.sig_loss_record.label.set_value(
					input_data_eval[3].get(), unsafe=True)
				self.model.trainer_extra_schema.tanh_loss_record.label.set_value(
					input_data_eval[4].get(), unsafe=True)

			eval_net = instantiator.generate_eval_net(self.model)
			workspace.CreateNet(eval_net)
			self.net_store['eval_net'] = eval_net


	def train_with_eval(
		self,
		num_epoch=1,
		report_interval=0,
		eval_during_training=False,
	):
		''' Fastest mode: report_interval = 0
			Medium mode: report_interval > 0, eval_during_training=False
			Slowest mode: report_interval > 0, eval_during_training=True
		'''
		num_batch_per_epoch = int(
			self.input_data_store['train'][1] / 
			self.batch_size
		)
		if not self.input_data_store['train'][1] % self.batch_size == 0:
			num_batch_per_epoch += 1
			print('[Warning]: batch_size cannot be divided. ' + 
				'Run on {} example instead of {}'.format(
						num_batch_per_epoch * self.batch_size,
						self.input_data_store['train'][1]
					)
				)
		print('<<< Run {} iteration'.format(num_epoch * num_batch_per_epoch))

		train_net = self.net_store['train_net']
		if report_interval > 0:
			print('>>> Training with Reports')
			num_eval = int(num_epoch / report_interval)
			num_unit_iter = int((num_batch_per_epoch * num_epoch)/num_eval)
			if eval_during_training and 'eval_net' in self.net_store:
				print('>>> Training with Eval Reports (Slowest mode)')
				eval_net = self.net_store['eval_net']
			for i in range(num_eval):
				workspace.RunNet(
					train_net.Proto().name, 
					num_iter=num_unit_iter
				)
				self.reports['epoch'].append((i + 1) * report_interval)
				train_loss = np.asscalar(schema.FetchRecord(self.loss).get())
				self.reports['train_loss'].append(train_loss)
				# Add metrics
				train_l1_metric = np.asscalar(schema.FetchRecord(
					self.model.metrics_schema.l1_metric).get())
				self.reports['train_l1_metric'].append(train_l1_metric)
				train_scaled_l1_metric = np.asscalar(schema.FetchRecord(
					self.model.metrics_schema.scaled_l1_metric).get())
				self.reports['train_scaled_l1_metric'].append(
					train_scaled_l1_metric)

				if eval_during_training and 'eval_net' in self.net_store:
					workspace.RunNet(
						eval_net.Proto().name,
						num_iter=num_unit_iter)
					eval_loss = np.asscalar(schema.FetchRecord(self.loss).get())
					# Add metrics
					self.reports['eval_loss'].append(eval_loss)
					eval_l1_metric = np.asscalar(schema.FetchRecord(
						self.model.metrics_schema.l1_metric).get())
					self.reports['eval_l1_metric'].append(eval_l1_metric)
					eval_scaled_l1_metric = np.asscalar(schema.FetchRecord(
						self.model.metrics_schema.scaled_l1_metric).get())
					self.reports['eval_scaled_l1_metric'].append(
						eval_scaled_l1_metric)
		else:
			print('>>> Training without Reports (Fastest mode)')
			workspace.RunNet(
				train_net, 
				num_iter=num_epoch * num_batch_per_epoch,
			)
			
		print('>>> Saving test model')

		# Save Net
		exporter.save_net(
			self.net_store['pred_net'], 
			self.model, 
			self.model_name+'_init', self.model_name+'_predict'
		)

		# Save Loss Trend
		if report_interval > 0:
			self.save_loss_trend(self.model_name)



	def draw_nets(self):
		for net_name in self.net_store:
			net = self.net_store[net_name]
			graph = net_drawer.GetPydotGraph(net.Proto().op, rankdir='TB')
			with open(self.model_name + '_' + net.Name() + ".png",'wb') as f:
				f.write(graph.create_png())
			with open(self.model_name + '_' + net.Name() + "_proto.txt",'wb') as f:
				f.write(str(net.Proto()))
				

	def predict_ids(self, vg, vd):
		# preproc the input
		vg = vg.astype(np.float32)
		vd = vd.astype(np.float32)
		if len(self.preproc_param) == 0:
			self.preproc_param = pickle.load(
				open(self.pickle_file_name, "rb" )
			)
		dummy_ids = np.zeros(len(vg))
		preproc_data_arrays = [vg, vd, dummy_ids]
		if self.train_target == TrainTarget.ORIGIN: 
			preproc_data_arrays = preproc.dc_iv_preproc(
				vg, vd, dummy_ids, 
				self.preproc_param['scale'], 
				self.preproc_param['vg_shift'], 
			)

		_preproc_data_arrays = [np.expand_dims(
			x, axis=1)  if x.ndim == 1 else x for x in preproc_data_arrays]
		workspace.FeedBlob('DBInput_train/sig_input', _preproc_data_arrays[0])
		workspace.FeedBlob('DBInput_train/tanh_input', _preproc_data_arrays[1])

		if self.train_target == TrainTarget.ADJOINT:
			adjoint_input = np.ones((vg.shape[0], 1))
			workspace.FeedBlob('DBInput_train/adjoint_input', adjoint_input)

		pred_net = self.net_store['pred_net']
		workspace.RunNet(pred_net)

		_ids = np.squeeze(schema.FetchRecord(self.pred).get())
		ids = _ids
		if self.train_target == TrainTarget.ORIGIN: 
			restore_id_func, _ = preproc.get_restore_id_func( 
				self.preproc_param['scale'], 
				self.preproc_param['vg_shift'], 
			)
			ids = restore_id_func(_ids)
			
		return _ids, ids

	def plot_loss_trend(self):
		plt.plot(
			self.reports['epoch'], 
			self.reports['train_loss'], 'r', 
			label='train error'
		)
		plt.plot(
			self.reports['epoch'], 
			self.reports['train_scaled_l1_metric'], 'b', 
			label='train_scaled_l1_metric'
		)
		plt.plot(
			self.reports['epoch'], 
			self.reports['train_l1_metric'], 'g', 
			label='train_l1_metric'
		)
		if len(self.reports['eval_loss']) > 0:
			plt.plot(
				self.reports['epoch'], 
				self.reports['eval_loss'], 'r--',
				label='eval error'
			)
			plt.plot(
				self.reports['epoch'], 
				self.reports['eval_scaled_l1_metric'], 'b--',
				label='eval_scaled_l1_metric'
			)
			plt.plot(
				self.reports['epoch'], 
				self.reports['eval_l1_metric'], 'g--', 
				label='eval_l1_metric'
			)
		plt.legend()
		plt.show()

	def save_loss_trend(self,save_name):
		if len(self.reports['eval_loss'])>0:
			f = open(save_name+'_loss_trend.csv', "w")
			f.write(
				"{},{},{},{},{},{},{}\n".format(
					"epoch", "train_loss","eval_loss","train_l1_metric",
					"eval_l1_metric","train_scaled_l1_metric","eval_scaled_l1_metric"))
			for x in zip(
				self.reports['epoch'],self.reports['train_loss'],
				self.reports['eval_loss'],self.reports['train_l1_metric'],
				self.reports['eval_l1_metric'],
				self.reports['train_scaled_l1_metric'],
				self.reports['eval_scaled_l1_metric'] ):
				f.write("{},{},{},{},{},{},{}\n".format(
					x[0], x[1], x[2], x[3], x[4], x[5], x[6]))
			f.close()
		else:
			f = open(save_name+'_loss_trend.csv', "w")
			f.write("{},{},{},{}\n".format("epoch", "train_loss","train_l1_metric",
									 "train_scaled_l1_metric" ))
			for x in zip(
				self.reports['epoch'],
				self.reports['train_loss'],self.reports['train_l1_metric'],
				self.reports['train_scaled_l1_metric'] ):
				f.write("{},{},{},{}\n".format(x[0], x[1], x[2], x[3]))
			f.close()

	
# --------------------------------------------------------
# ----------------   Global functions  -------------------
# --------------------------------------------------------

def predict_ids_grads(model_name, vg, vd):
	workspace.ResetWorkspace()

	# preproc the input
	vg = vg.astype(np.float32)
	vd = vd.astype(np.float32)
	#if len(self.preproc_param) == 0:
	preproc_param = pickle.load(
			open(model_name+'_preproc_param.p', "rb" )
		)
	dummy_ids = np.zeros(len(vg))
	preproc_data_arrays = preproc.dc_iv_preproc(
		vg, vd, dummy_ids, 
		preproc_param['scale'], 
		preproc_param['vg_shift'], 
	)
	_preproc_data_arrays = [np.expand_dims(
		x, axis=1) for x in preproc_data_arrays]

	workspace.FeedBlob('DBInput_train/sig_input', _preproc_data_arrays[0])
	workspace.FeedBlob('DBInput_train/tanh_input', _preproc_data_arrays[1])
	adjoint_input = np.ones((_preproc_data_arrays[0].shape[0], 1))
	workspace.FeedBlob('adjoint_input', adjoint_input)
	pred_net = exporter.load_net(model_name+'_init', model_name+'_predict')

	workspace.RunNet(pred_net)

	_ids = np.squeeze(workspace.FetchBlob('origin/Mul/origin_pred'))
	_sig_grad = np.squeeze(workspace.FetchBlob('adjoint/sig_fc_layer_0/output'))
	_tanh_grad = np.squeeze(workspace.FetchBlob('adjoint/tanh_fc_layer_0/output'))

	restore_id_func, get_restore_id_grad_func = preproc.get_restore_id_func( 
		preproc_param['scale']
	)
	ids = restore_id_func(_ids)
	sig_grad, tanh_grad = get_restore_id_grad_func(_sig_grad, _tanh_grad)
	return ids, sig_grad, tanh_grad

def plot_iv( 
	vg, vd, ids, 
	vg_comp = None, vd_comp = None, ids_comp = None,
	save_name = '',
	styles = ['vg_major_linear', 'vd_major_linear', 'vg_major_log', 'vd_major_log'],
	yLabel='I$_d$'
):
	if 'vg_major_linear' in styles:
		visualizer.plot_linear_Id_vs_Vd_at_Vg(
			vg, vd, ids, 
			vg_comp = vg_comp, vd_comp = vd_comp, ids_comp = ids_comp,
			save_name = save_name + 'vg_major_linear',
			yLabel=yLabel
		)
	if 'vd_major_linear' in styles:
		visualizer.plot_linear_Id_vs_Vg_at_Vd(
			vg, vd, ids, 
			vg_comp = vg_comp, vd_comp = vd_comp, ids_comp = ids_comp,
			save_name = save_name + 'vd_major_linear',
			yLabel=yLabel
		)
	if 'vg_major_log' in styles:
		visualizer.plot_log_Id_vs_Vd_at_Vg(
			vg, vd, ids, 
			vg_comp = vg_comp, vd_comp = vd_comp, ids_comp = ids_comp,
			save_name = save_name + 'vg_major_log',
			yLabel=yLabel
		)
	if 'vd_major_log' in styles:
		visualizer.plot_log_Id_vs_Vg_at_Vd(
			vg, vd, ids, 
			vg_comp = vg_comp, vd_comp = vd_comp, ids_comp = ids_comp,
			save_name = save_name + 'vd_major_log',
			yLabel=yLabel
		)

def _build_optimizer(optim_method, optim_param):
	if optim_method == 'AdaGrad':
		optim = optimizer.AdagradOptimizer(**optim_param)
	elif optim_method == 'SgdOptimizer':
		optim = optimizer.SgdOptimizer(**optim_param)
	elif optim_method == 'Adam':
		optim = optimizer.AdamOptimizer(**optim_param)
	else:
		raise Exception(
			'Did you foget to implement {}?'.format(optim_method))
	return optim


			
			
