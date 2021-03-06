�	����UW@����UW@!����UW@	�bZ'��@�bZ'��@!�bZ'��@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6����UW@�����S@1��Ɋ��@A&��)��?Iw|��!!@Y��_>�@*	���MB%�@2F
Iterator::Model��n�A@!M<?�UJ@)^�/�@1�4�1��I@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap@KW����?!��UzjG@)@j'�{�?1T��HG@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���d�z�?!C�\���?)�-;�?l�?1�T�ģ8�?:Preprocessing2U
Iterator::Model::ParallelMapV2���5�e�?!<�T�T�?)���5�e�?1<�T�T�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate@�P�%�?!�'�Ï��?)c�D(b�?1�� 90I�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicekg{�?!;�N���?)kg{�?1;�N���?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��6S!��?!�����G@)��iܛ߀?1�T�ȃ�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��l#~?!��KaV��?)��l#~?1��KaV��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 84.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�bZ'��@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�����S@�����S@!�����S@      ��!       "	��Ɋ��@��Ɋ��@!��Ɋ��@*      ��!       2	&��)��?&��)��?!&��)��?:	w|��!!@w|��!!@!w|��!!@B      ��!       J	��_>�@��_>�@!��_>�@R      ��!       Z	��_>�@��_>�@!��_>�@JGPUY�bZ'��@b �"N
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop�6�{�?!�6�{�?"&
CudnnRNNCudnnRNN�p��?!p}w����?";
gradients/split_1_grad/concatConcatV2��Q��?!7�yݰ��?"9
gradients/split_grad/concatConcatV2';�à�?!ӡ6�3�?"W
6gradient_tape/sequential_1/dense_5/BiasAdd/BiasAddGradBiasAddGrad���?!�2{AT�?"*
transpose_9	Transpose����4|?!&� 媌�?"(

concat_1_0ConcatV2�n۫�z?!3��<B��?"C
$gradients/transpose_9_grad/transpose	Transpose��Ϣby?!2]W���?"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGradx�Tę�w?!#��$�?"*
transpose_0	TransposeV�?�w?!N+�4�T�?Q      Y@Yz�t�1�@aI����W@q��;�PGE@y��\��?"�
both�Your program is POTENTIALLY input-bound because 84.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�42.5572% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 