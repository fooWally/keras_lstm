	{�L�x�,@{�L�x�,@!{�L�x�,@	���AH�@���AH�@!���AH�@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6{�L�x�,@����B��?1j�L�:@AAF@�#H�?IoB@�L!@YۆQ<��?*	F���Ԡ[@2F
Iterator::Model��ؖg�?!�I�MwrF@)v?T1�?1UyF���@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat_�Qڛ?!O�̜8@)��;��ؖ?1�Se04@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateh�4�;�?!<�θ�1@)�FN���?1�2"�N(@:Preprocessing2U
Iterator::Model::ParallelMapV2bg
�׈?!�@�P��%@)bg
�׈?1�@�P��%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip{ܷZ'.�?!p�z���K@)D�l����?1�|Y��7@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�ɐc�y?!D��r�@)�ɐc�y?1D��r�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�k����?!����P�6@)0�r.�u?1\�Wt�<@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�I�pt?!��SI)�@)�I�pt?1��SI)�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 10.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�60.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���AH�@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����B��?����B��?!����B��?      ��!       "	j�L�:@j�L�:@!j�L�:@*      ��!       2	AF@�#H�?AF@�#H�?!AF@�#H�?:	oB@�L!@oB@�L!@!oB@�L!@B      ��!       J	ۆQ<��?ۆQ<��?!ۆQ<��?R      ��!       Z	ۆQ<��?ۆQ<��?!ۆQ<��?JGPUY���AH�@b 