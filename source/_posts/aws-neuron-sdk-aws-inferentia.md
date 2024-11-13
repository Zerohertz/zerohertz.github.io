---
title: AWS Neuron SDK & AWS Inferentia
date: 2023-05-12 10:36:33
categories:
- 4. MLOps
tags:
- AWS
- Docker
- Triton Inference Server
- Python
---
# Introduction

Model serving 시 모델의 입력에 대해 출력을 산출하는 과정인 [순전파 (forward propagation)](https://076923.github.io/posts/AI-5/#%EC%88%9C%EC%A0%84%ED%8C%8Cforward-propagation)는 model 내부의 layer들을 거치는데, 각 layer에는 수많은 단순 연산들이 존재한다.
이를 병렬로 수행하기 위해 CPU (Central Processing Unit)보다 병렬 계산에 이점이 있는 GPU (Grphic Processing Unit)을 많이 활용하고 있다.
하지만 아래와 같은 한계점이 존재한다.

+ 비싸다. (물리적 GPU 비용, 전력 소모, 클라우드 서비스 등 모든 측면에서)
+ 물리 코어를 가상 코어로 나눠 사용할 수 없어 자원을 낭비하기 쉽다.
+ 연산량이 적은 모델에 대해 추론할 때 CPU와 GPU 간 메모리 통신 오버헤드가 큰 경우 CPU에서 자체적 처리가 더 효율적일 수 있다.

이러한 이유들로 다양한 회사에서 GPU에 비해 저렴하지만 효율적인 ML 연산을 할 수 있는 AI 가속기들을 아래와 같이 개발하고 있다.

+ AWS Trainium, Inferentia
+ Google TPU (Tensor Processing Unit)
+ Apple ANE (Apple Neural Engine)

<!-- More -->

AI 가속기들은 ASIC (Application-Specific Integrated Circuit, 특정 용도용 집적 회로) 기반으로 개발되고 있다.
ASIC은 공정상의 이유로 소량 생산 시 생산 비용이 비싸고 설계 및 수정도 어려운 한계점들이 있지만, ML 연산은 대부분이 단순하고 수많은 독립적 연산을 처리하는 패턴이기 때문에 model serving을 위한 하드웨어를 생산하기 적합하다.
AI 가속기를 사용한 Amazon EC2 객체는 아래와 같다.

+ Amazon EC2 Trn1/Trn1n
  + AWS Trainium으로 구동
  + 모델 학습에 최적화 (학습 시간 단축, 저렴한 비용)
+ Amazon EC2 Inf1/Inf2
  + AWS Inferentia로 구동
  + 모델 추론에 최적화 (저렴한 비용, 높은 처리량, 낮은 지연 시간)

두 인스턴스 중 모델을 serving 할 수 있는 Amazon EC2 Inf* 인스턴스에 대해 알아보겠다.
Amazon EC2 Inf*는 앞서 설명한 ASIC으로 구동되기 때문에 모델을 AWS Neuron SDK로 모델을 컴파일해야한다.

+ AWS Neuron: AWS Trainium 및 AWS Inferentia 기반 인스턴스에서 deep learning workloads를 실행하는데 사용되는 SDK
  + End-to-End ML development lifecycle에서 새로운 모델 구축, 학습 및 최적화 후 배포를 위해 사용
  + TensorFlow, PyTorch, Apache MXNet에 통합되는 deep learning compiler, runtime, tools 포함

AWS Neuron SDK를 통한 모델 컴파일은 [TensorRT 변환](https://zerohertz.github.io/how-to-convert-a-pytorch-model-to-tensorrt/)과 유사하게 진행된다.
변환된 모델을 Triton Inference Server로 모델을 배포하면 모델 serving에 대한 준비는 모두 완료된다.

---

# Model Compile

앞서 말한 것과 같이 AWS Neuron SDK를 통해 모델을 컴파일하기 위해서는 모델의 구조를 담고 있는 deep learning framework (PyTorch, TensorFlow, ...) 기반의 코드와 학습이 완료된 가중치를 준비해야한다.

> AWS Neuron SDK 설치

```shell Install AWS Neuron SDK for Inf1
$ pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
$ pip install torch-neuron neuron-cc[tensorflow] "protobuf" torchvision
```

```shell Install AWS Neuron SDK for Inf2
$ sudo apt-get install aws-neuronx-collectives=2.* -y
$ sudo apt-get install aws-neuronx-runtime-lib=2.* -y
$ pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
$ pip install neuronx-cc==2.* torch-neuronx torchvision
```

> AWS Neuron SDK 기반 모델 컴파일

```python PyTorch2Neuron for Inf1
import torch
import torch.neuron

pretrained_dict = torch.load(weightfile, map_location=torch.device('cpu'))
model.load_state_dict(pretrained_dict)
model.eval()

input_tensor = torch.zeros([B, C, H, W], dtype=torch.float32)
model_neuron = torch.neuron.trace(model, [input_tensor])

filename = 'model_neuron.pt'
torch.jit.save(model_neuron, filename)
model_neuron.save(filename)
```

+ 컴파일은 [`torch.neuron.trace()`](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuron/api-compilation-python-api.html?highlight=torch.neuron.trace#torch_neuron.trace) 함수를 사용해 진행할 수 있다.

> 추론을 통한 결과 비교

```python Inference.py
model_neuron = torch.jit.load('model_neuron.pt')

output_cpu = model(input_tensor)
output_neuron = model_neuron(input_tensor)

print("Results of CPU: ", output_cpu)
print("Results of Neuron: ", output_neuron)
```

이를 응용하고 시험하기 위해 STD 모델인 [CRAFT](https://github.com/Zerohertz/CRAFT/tree/Neuron)를 AWS Neuron SDK로 컴파일하기 위해 아래 코드를 개발했다.

```python torch2neuron.py
import torch
import torch.neuron
import cv2
import numpy as np

from config.load_config import load_yaml, DotDict
from model.craft import CRAFT
from utils.util import copyStateDict
from utils.craft_utils import getDetBoxes
from data import imgproc


if __name__ == "__main__":
    config = load_yaml("main")
    config = DotDict(config)

    model = CRAFT()
    model_path = config.test.trained_model
    net_param = torch.load(model_path)
    model.load_state_dict(copyStateDict(net_param["craft"]))
    model.eval()

    image_path = config.test.custom_data.test_data_dir
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        img, config.test.custom_data.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=config.test.custom_data.mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio
    input_tensor = imgproc.normalizeMeanVariance(img_resized)
    input_tensor = torch.from_numpy(input_tensor).permute(2,0,1)
    input_tensor = torch.autograd.Variable(input_tensor.unsqueeze(0))
    print(input_tensor.shape)

    with torch.no_grad():
        y_cpu, feature_cpu = model(input_tensor)
    score_text = y_cpu[0, :, :, 0].cpu().data.numpy().astype(np.float32)
    score_link = y_cpu[0, :, :, 1].cpu().data.numpy().astype(np.float32)
    score_text_cpu = score_text[: size_heatmap[0], : size_heatmap[1]]
    score_link_cpu = score_link[: size_heatmap[0], : size_heatmap[1]]

    boxes_cpu, polys_cpu = getDetBoxes(
        score_text_cpu, score_link_cpu,
        config.test.custom_data.text_threshold,
        config.test.custom_data.link_threshold,
        config.test.custom_data.low_text,
        config.test.custom_data.poly
    )

    convert_neuron = True
    if convert_neuron:
        model_neuron = torch.neuron.trace(model, [input_tensor], compiler_workdir='./logs')
        filename = 'model_neuron.pt'
        model_neuron.save(filename)

    validate_neuron = False
    if validate_neuron:
        model_neuron = torch.jit.load('model_neuron.pt')
        y_neuron, feature_neuron = model_neuron(input_tensor)
        score_text = y_neuron[0, :, :, 0].cpu().data.numpy().astype(np.float32)
        score_link = y_neuron[0, :, :, 1].cpu().data.numpy().astype(np.float32)
        score_text_neuron = score_text[: size_heatmap[0], : size_heatmap[1]]
        score_link_neuron = score_link[: size_heatmap[0], : size_heatmap[1]]
        boxes_neuron, polys_neuron = getDetBoxes(
            score_text_neuron, score_link_neuron,
            config.test.custom_data.text_threshold,
            config.test.custom_data.link_threshold,
            config.test.custom_data.low_text,
            config.test.custom_data.poly
        )
        print(score_text_cpu)
        print(score_text_neuron)
        print(score_link_cpu)
        print(score_link_neuron)
        print(boxes_cpu)
        print(boxes_neuron)
```

```shell Logs
$ python torch2neuron.py 
torch.Size([1, 3, 2400, 2400])
INFO:Neuron:All operators are compiled by neuron-cc (this does not guarantee that neuron-cc will successfully compile)
INFO:Neuron:Number of arithmetic operators (pre-compilation) before = 95, fused = 95, percent fused = 100.0%
INFO:Neuron:Compiling function _NeuronGraph$278 with neuron-cc; log file is at /home/jovyan/local/1_user/hgoh@agilesoda.ai/CRAFT/logs/0/graph_def.neuron-cc.log
INFO:Neuron:Compiling with command line: '/opt/conda/envs/neuron/bin/neuron-cc compile /home/jovyan/local/1_user/hgoh@agilesoda.ai/CRAFT/logs/0/graph_def.pb --framework TENSORFLOW --pipeline compile SaveTemps --output /home/jovyan/local/1_user/hgoh@agilesoda.ai/CRAFT/logs/0/graph_def.neff --io-config {"inputs": {"0:0": [[1, 3, 2400, 2400], "float32"]}, "outputs": ["aten_permute/transpose:0", "double_conv_59/Sequential_1/BatchNorm2d_10/aten_relu/Relu:0"]} --verbose 35'
INFO:Neuron:Number of arithmetic operators (post-compilation) before = 95, compiled = 95, percent compiled = 100.0%
INFO:Neuron:The neuron partitioner created 1 sub-graphs
INFO:Neuron:Neuron successfully compiled 1 sub-graphs, Total fused subgraphs = 1, Percent of model sub-graphs successfully compiled = 100.0%
INFO:Neuron:Compiled these operators (and operator counts) to Neuron:
INFO:Neuron: => aten::Int: 6
INFO:Neuron: => aten::_convolution: 27
INFO:Neuron: => aten::batch_norm: 20
INFO:Neuron: => aten::cat: 4
INFO:Neuron: => aten::max_pool2d: 5
INFO:Neuron: => aten::permute: 1
INFO:Neuron: => aten::relu_: 23
INFO:Neuron: => aten::size: 6
INFO:Neuron: => aten::upsample_bilinear2d: 3
```

상세한 로그들은 [여기](https://github.com/Zerohertz/CRAFT/tree/Neuron/logs/0)에서 확인할 수 있다.

+ 결과 비교를 위해서는 Amazon EC2 Inf1 인스턴스를 실행해야하지만 아직은 불가능하기 때문에 일단 여기까지! $\rightarrow$ 차후 `model_neuron.pt`를 Python backend 기반 Triton Inference Server로 모델 배포 가능!
+ 모델 컴파일 시 [Dynamic Shapes](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.10.0/general/arch/neuron-features/dynamic-shapes.html)는 현재까지는 지원 X

하지만 동적인 입력을 위해 아래와 같이 입력 텐서의 크기를 `torch.nn.functional.pad`로 조정해줄 수 있다.

```python
import torch


p = torch.nn.functional.pad
ph, pw = max(0, 2400 - input_tensor.shape[2]), max(0, 2400 - input_tensor.shape[3])
input_tensor = p(input_tensor, (0,pw,0,ph))
```

모든 케이스에 적용이 가능하진 않지만 CNN (Convolution Neural Network)과 같은 계층으로 구성된 신경망에 대해서는 가능하다.

---

# Compile 외않되?

위에서 진행한 것과 같이 Amazon EC2 Inf1 인스턴스에 모델들을 서빙하기위해 AWS Neuron SDK로 모델들을 컴파일하는 도중 STR 모델인 [PARSeq](https://github.com/baudm/parseq)가 컴파일 시 에러가 발생함을 확인하였다.
컴파일 에러에 대한 ChatGPT 선생님의 소견은 아래와 같다.

> 이는 PyTorch의 동적 그래프 실행 방식 때문입니다. PyTorch는 실행 시에 모델의 형태를 동적으로 조정할 수 있기 때문에 오류가 발생하지 않을 수 있습니다. 그러나 Neuron은 컴파일 시에 정적인 그래프를 사용하므로 텐서 형태의 일치가 필요합니다. 그래서 Neuron 변환 과정에서 오류가 발생하는 것입니다.
> 이러한 문제를 해결하기 위해서는 모델 변환 과정에서 `aten::view` 연산을 수정해야 합니다. 현재 모델 변환에 사용된 `aten::view` 연산에서 잘못된 형태로 텐서를 변경하려고 하기 때문에 오류가 발생한 것입니다. `aten::view` 연산을 수정하여 입력 텐서의 형태를 목표 형태로 올바르게 변경하도록 해야 합니다.

선생님의 말씀과 같이 Python만을 사용해 동일 모델을 순전파하여 추론하였을 때는 에러가 발생하지 않음을 확인할 수 있었고, 이는 내부에서 [동적인 텐서 크기](https://github.com/baudm/parseq/blob/main/.github/system.png)가 존재하여 발생하였음을 추측할 수 있었다.
자세하게는 특정 레이어에서 텐서가 순전파될 때 텐서의 크기가 동적이기 때문에 발생하였음을 하단의 Error Logs를 살펴보면 알 수 있다.

<details>
<summary>
Error Logs
</summary>

```bash
/opt/conda/envs/neuron/lib/python3.8/site-packages/torch/__init__.py:853: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert condition, message
INFO:Neuron:There are 105 ops of 2 different types in the TorchScript that are not compiled by neuron-cc: aten::embedding, aten::index_put_, (For more information see https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/compiler/neuron-cc/neuron-cc-ops/neuron-cc-ops-pytorch.html)
INFO:Neuron:Number of arithmetic operators (pre-compilation) before = 8878, fused = 8295, percent fused = 93.43%
WARNING:tensorflow:From /opt/conda/envs/neuron/lib/python3.8/site-packages/torch_neuron/ops/aten.py:2387: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
WARNING:tensorflow:From /opt/conda/envs/neuron/lib/python3.8/site-packages/torch_neuron/ops/aten.py:3866: The name tf.matrix_band_part is deprecated. Please use tf.linalg.band_part instead.

INFO:Neuron:PyTorch to TF conversion failed to resolve function on aten::view with inputs [<tf.Tensor 'Decoder_132/DecoderLayer_3/MultiheadAttention_14/aten_chunk/split:0' shape=(2, 1, 384) dtype=float32>, [1, 12, 32]]
INFO:Neuron:Exception = Cannot reshape a tensor with 768 elements to shape [1,12,32] (384 elements) for 'Decoder_132/DecoderLayer_3/MultiheadAttention_14/aten_view_1/Reshape' (op: 'Reshape') with input shapes: [2,1,384], [3] and with input tensors computed as partial shapes: input[1] = [1,12,32].
WARNING:Neuron:torch.neuron.trace failed on _NeuronGraph$4187; falling back to native python function call
ERROR:Neuron:Cannot reshape a tensor with 768 elements to shape [1,12,32] (384 elements) for 'Decoder_132/DecoderLayer_3/MultiheadAttention_14/aten_view_1/Reshape' (op: 'Reshape') with input shapes: [2,1,384], [3] and with input tensors computed as partial shapes: input[1] = [1,12,32].
Traceback (most recent call last):
  File "/opt/conda/envs/neuron/lib/python3.8/site-packages/tensorflow_core/python/framework/ops.py", line 1607, in _create_c_op
    c_op = c_api.TF_FinishOperation(op_desc)
tensorflow.python.framework.errors_impl.InvalidArgumentError: Cannot reshape a tensor with 768 elements to shape [1,12,32] (384 elements) for 'Decoder_132/DecoderLayer_3/MultiheadAttention_14/aten_view_1/Reshape' (op: 'Reshape') with input shapes: [2,1,384], [3] and with input tensors computed as partial shapes: input[1] = [1,12,32].

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/opt/conda/envs/neuron/lib/python3.8/site-packages/torch_neuron/convert.py", line 413, in op_converter
    neuron_function = self.subgraph_compiler(
  File "/opt/conda/envs/neuron/lib/python3.8/site-packages/torch_neuron/decorators.py", line 81, in trace
    transform_torch_graph_to_tensorflow(jit_trace, example_inputs, separate_weights=separate_weights, neuron_graph=func, **kwargs)
  File "/opt/conda/envs/neuron/lib/python3.8/site-packages/torch_neuron/decorators.py", line 634, in transform_torch_graph_to_tensorflow
    raise e
  File "/opt/conda/envs/neuron/lib/python3.8/site-packages/torch_neuron/decorators.py", line 628, in transform_torch_graph_to_tensorflow
    tensor_outputs = local_func(op, *tensor_inputs)
  File "/opt/conda/envs/neuron/lib/python3.8/site-packages/torch_neuron/ops/aten.py", line 2540, in view
    return reshape(op, tensor, shape)
  File "/opt/conda/envs/neuron/lib/python3.8/site-packages/torch_neuron/ops/aten.py", line 2548, in reshape
    out = tf.reshape(tensor, shape)
  File "/opt/conda/envs/neuron/lib/python3.8/site-packages/tensorflow_core/python/ops/array_ops.py", line 131, in reshape
    result = gen_array_ops.reshape(tensor, shape, name)
  File "/opt/conda/envs/neuron/lib/python3.8/site-packages/tensorflow_core/python/ops/gen_array_ops.py", line 8114, in reshape
    _, _, _op = _op_def_lib._apply_op_helper(
  File "/opt/conda/envs/neuron/lib/python3.8/site-packages/tensorflow_core/python/framework/op_def_library.py", line 792, in _apply_op_helper
    op = g.create_op(op_type_name, inputs, dtypes=None, name=scope,
  File "/opt/conda/envs/neuron/lib/python3.8/site-packages/tensorflow_core/python/util/deprecation.py", line 507, in new_func
    return func(*args, **kwargs)
  File "/opt/conda/envs/neuron/lib/python3.8/site-packages/tensorflow_core/python/framework/ops.py", line 3356, in create_op
    return self._create_op_internal(op_type, inputs, dtypes, input_types, name,
  File "/opt/conda/envs/neuron/lib/python3.8/site-packages/tensorflow_core/python/framework/ops.py", line 3418, in _create_op_internal
    ret = Operation(
  File "/opt/conda/envs/neuron/lib/python3.8/site-packages/tensorflow_core/python/framework/ops.py", line 1769, in __init__
    self._c_op = _create_c_op(self._graph, node_def, grouped_inputs,
  File "/opt/conda/envs/neuron/lib/python3.8/site-packages/tensorflow_core/python/framework/ops.py", line 1610, in _create_c_op
    raise ValueError(str(e))
ValueError: Cannot reshape a tensor with 768 elements to shape [1,12,32] (384 elements) for 'Decoder_132/DecoderLayer_3/MultiheadAttention_14/aten_view_1/Reshape' (op: 'Reshape') with input shapes: [2,1,384], [3] and with input tensors computed as partial shapes: input[1] = [1,12,32].
INFO:Neuron:Number of arithmetic operators (post-compilation) before = 8878, compiled = 0, percent compiled = 0.0%
INFO:Neuron:The neuron partitioner created 1 sub-graphs
INFO:Neuron:Neuron successfully compiled 0 sub-graphs, Total fused subgraphs = 1, Percent of model sub-graphs successfully compiled = 0.0%
INFO:Neuron:Compiled these operators (and operator counts) to Neuron:
INFO:Neuron:Not compiled operators (and operator counts) to Neuron:
INFO:Neuron: => aten::Int: 1992 [supported]
INFO:Neuron: => aten::_convolution: 1 [supported]
INFO:Neuron: => aten::add: 233 [supported]
INFO:Neuron: => aten::argmax: 51 [supported]
INFO:Neuron: => aten::baddbmm: 52 [supported]
INFO:Neuron: => aten::bmm: 156 [supported]
INFO:Neuron: => aten::cat: 54 [supported]
INFO:Neuron: => aten::chunk: 104 [supported]
INFO:Neuron: => aten::contiguous: 416 [supported]
INFO:Neuron: => aten::cumsum: 1 [supported]
INFO:Neuron: => aten::div: 208 [supported]
INFO:Neuron: => aten::dropout: 361 [supported]
INFO:Neuron: => aten::embedding: 104 [not supported]
INFO:Neuron: => aten::eq: 1 [supported]
INFO:Neuron: => aten::expand: 2 [supported]
INFO:Neuron: => aten::fill_: 51 [supported]
INFO:Neuron: => aten::flatten: 1 [supported]
INFO:Neuron: => aten::full: 3 [supported]
INFO:Neuron: => aten::gelu: 64 [supported]
INFO:Neuron: => aten::gt: 1 [supported]
INFO:Neuron: => aten::index_put_: 1 [not supported]
INFO:Neuron: => aten::layer_norm: 285 [supported]
INFO:Neuron: => aten::linear: 516 [supported]
INFO:Neuron: => aten::masked_fill: 1 [supported]
INFO:Neuron: => aten::matmul: 24 [supported]
INFO:Neuron: => aten::mul: 741 [supported]
INFO:Neuron: => aten::ones: 1 [supported]
INFO:Neuron: => aten::permute: 12 [supported]
INFO:Neuron: => aten::reshape: 25 [supported]
INFO:Neuron: => aten::select: 51 [supported]
INFO:Neuron: => aten::size: 819 [supported]
INFO:Neuron: => aten::slice: 675 [supported]
INFO:Neuron: => aten::softmax: 116 [supported]
INFO:Neuron: => aten::split_with_sizes: 208 [supported]
INFO:Neuron: => aten::squeeze: 50 [supported]
INFO:Neuron: => aten::sub: 52 [supported]
INFO:Neuron: => aten::to: 1 [supported]
INFO:Neuron: => aten::transpose: 857 [supported]
INFO:Neuron: => aten::triu: 2 [supported]
INFO:Neuron: => aten::unbind: 12 [supported]
INFO:Neuron: => aten::unsqueeze: 52 [supported]
INFO:Neuron: => aten::view: 521 [supported]
Traceback (most recent call last):
  File "torch2neuron.py", line 46, in <module>
    model_neuron = torch.neuron.trace(model, [input_tensor])
  File "/opt/conda/envs/neuron/lib/python3.8/site-packages/torch_neuron/convert.py", line 217, in trace
    cu.stats_post_compiler(neuron_graph)
  File "/opt/conda/envs/neuron/lib/python3.8/site-packages/torch_neuron/convert.py", line 530, in stats_post_compiler
    raise RuntimeError(
RuntimeError: No operations were successfully partitioned and compiled to neuron for this model - aborting trace!
```

</details>

[비슷한 구조의 Encoder-Decoder 모델의 컴파일 과정 예제](https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuron/inference/fairseq/Fairseq.ipynb)를 기반으로 컴파일 에러를 해결해보려 했으나,,, 위에서 기술한 것과 같이 동적 그래프 방식 모델은 AWS Neuron SDK로 컴파일이 불가능하다.
아래와 같이 `PARSeq.decode()` 메서드에 입력되는 텐서가 슬라이싱되어 입력되기 때문에 매 반복에 따라서 크기가 변하는 것을 확인할 수 있고, 이를 해결하기 위해 동적인 입력 텐서에 대해 가장 큰 차원의 입력 텐서를 기준으로 작은 텐서를 패딩하여 텐서의 차원을 고정하려 했으나 계층의 순전파 이후 결과가 변경되기 때문에 불가능한 것으로 결론지었다.

```python
...
class PARSeq(CrossEntropySystem):
    ...
    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        ...
        if self.decode_ar:
            ...
            for i in range(num_steps):
                j = i + 1  # next token index
                # Efficient decoding:
                # Input the context up to the ith token. We use only one query (at position = i) at a time.
                # This works because of the lookahead masking effect of the canonical (forward) AR context.
                # Past tokens have no access to future tokens, hence are fixed once computed.
                tgt_out = self.decode(tgt_in[:, :j], memory, tgt_mask[:j, :j], tgt_query=pos_queries[:, i:j],
                                      tgt_query_mask=query_mask[i:j, :j])
                # the next token probability is in the output's ith token position
                p_i = self.head(tgt_out)
                logits.append(p_i)
                if j < num_steps:
                    # greedy decode. add the next token index to the target input
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
                    # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                    if testing and (tgt_in == self.eos_id).any(dim=-1).all():
                        break
...
```

하지만 이러한 구조를 정적으로 변경하면 컴파일이 가능하다. (중요한 것은 꺾이지 않는 마음.)
모델의 성능은 감소하겠지만, Amazon EC2 Inf1의 비용이 매우 저렴하기 때문에 경우에 따라서 좋은 선택지가 될 수 있다.
모델을 정적으로 만들기 위해 아래와 같이 변경하였다.

```python
...
class PARSeq(nn.Module):
    ...
    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[Tensor] = None,
               tgt_padding_mask: Optional[Tensor] = None, tgt_query: Optional[Tensor] = None,
               tgt_query_mask: Optional[Tensor] = None):
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.dropout(null_ctx)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)
    ...
    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        max_length = 50
        bs = 1
        num_steps = max_length + 1
        memory = self.encode(images)
        pos_queries = self.pos_queries[:, :51].expand(1, -1, -1)
        tgt_in = torch.full((1, 1), self.bos_id, dtype=torch.long)
        tgt_out = self.decode(tgt_in, memory, tgt_query=pos_queries)
        logits = self.head(tgt_out)
        return logits
```

이렇게 정적으로 변경한 모델을 AWS Neuron SDK로 컴파일하는 코드는 아래와 같다.

```python torch2neuron.py
import torch
import torch.neuron
from torchvision import transforms as T
import cv2
from PIL import Image
import numpy as np

from model.parseq import PARSeq


def get_transform(img_size):
    transforms = []
    transforms.extend([
        T.Resize(img_size, T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5)
    ])
    return T.Compose(transforms)

if __name__ == "__main__":
    loaded_dict = torch.load('parseq.pth')
    config = loaded_dict['cfg']
    model = PARSeq(**config.model)
    if 'model' in loaded_dict.keys():
        model_state_dict = loaded_dict['model']
    model.load_state_dict(model_state_dict)
    model.eval()

    img_transform = get_transform(config.data.img_size)
    img_path = 'data/target.jpg'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.dstack([img, img, img])
    img = Image.fromarray(img)
    input_tensor = img_transform(img).unsqueeze(0)

    with torch.no_grad():
        p_cpu = model(input_tensor).softmax(-1)

    preds_cpu, probs_cpu = model.tokenizer.decode(p_cpu)

    convert_neuron = True
    if convert_neuron:
        ops = torch.neuron.get_supported_operations() + ['aten::embedding']
        print(torch.neuron.analyze_model(model, [input_tensor]))
        model_neuron = torch.neuron.trace(model, [input_tensor], op_whitelist=ops, compiler_workdir='./logs')
        filename = 'model_neuron.pt'
        model_neuron.save(filename)

    validate_neuron = False
    if validate_neuron:
        model_neuron = torch.jit.load('model_neuron.pt')
        p_neuron = model_neuron(input_tensor).softmax(-1)
        preds_neuron, probs_neuron = model.tokenizer.decode(p_neuron)
        print(preds_cpu)
        print(probs_cpu)
        print(preds_neuron)
        print(probs_neuron)
```

CRAFT와 다르게 PARSeq는 모델 구조의 이해도가 상대적으로 떨어지기 때문에 컴파일에 어려움이 많았다.
MLOps 개발자의 길은 험난하다...

---

# Refereces

1. Hyperconnect: [머신러닝 모델 서빙 비용 1/4로 줄이기](https://hyperconnect.github.io/2022/12/13/infra-cost-optimization-with-aws-inferentia.html)
2. AWS: [Welcome to AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/)
   + [Inferentia1 Architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/inferentia.html#inferentia-architecture)
   + [Inferentia2 Architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/inferentia2.html#inferentia2-architecture)
   + [Model Architecture Fit Guidelines](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/model-architecture-fit.html#model-architecture-fit)
3. [Amazon EC2 Trn1](https://aws.amazon.com/ko/ec2/instance-types/trn1/)
4. [Amazon EC2 Inf2](https://aws.amazon.com/ko/ec2/instance-types/inf2/)
5. Scatter Lab
   + [AWS Inferentia를 이용한 모델 서빙 비용 최적화: 모델 서버 비용 2배 줄이기 1탄](https://tech.scatterlab.co.kr/aws-inferentia/)
   + [AWS Inferentia를 이용한 모델 서빙 비용 최적화: 모델 서버 비용 2배 줄이기 2탄](https://tech.scatterlab.co.kr/aws-inferentia-2/)