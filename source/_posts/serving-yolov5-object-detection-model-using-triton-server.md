---
title: Serving YOLOv5 Object Detection Model using Triton Inference Server
date: 2023-04-21 08:07:51
categories:
- 4. MLOps
tags:
- Docker
- Triton Inference Server
- TensorRT
- Python
---

# Introduction

> Definition of Triton Inference Server

[Triton Inference Server](https://developer.nvidia.com/triton-inference-server)는 NVIDIA에서 개발한 딥 러닝 모델 인퍼런스를 위한 고성능 인퍼런스 서버입니다. Triton Inference Server는 다중 모델을 지원하며, TensorFlow, PyTorch, ONNX 등의 주요 딥 러닝 프레임워크를 모두 지원합니다.
이를 통해 사용자는 다양한 모델을 효율적으로 서빙할 수 있습니다.
Triton Inference Server는 NVIDIA TensorRT 엔진을 기반으로하며, GPU 가속을 통해 모델 추론을 빠르게 수행할 수 있습니다.
또한 Triton Inference Server는 TensorFlow Serving과 호환되는 gRPC 인터페이스를 제공하며 Triton Inference Server는 TensorFlow Serving, TorchServe와 같은 기존 인퍼런스 서버와 비교하여 성능 및 유연성 면에서 우수한 성능을 발휘합니다.
Triton Inference Server는 Kubernetes, Docker 및 NVIDIA DeepOps와 같은 오케스트레이션 툴과 통합되어 쉽게 배포할 수 있습니다.
Triton Inference Server는 성능, 확장성 및 유연성 면에서 우수한 기능을 제공하므로, 대규모 딥 러닝 모델 인퍼런스를 위한 선택적이고 강력한 도구로 자리 잡고 있습니다.

> Docker Image: `triton-server`

Triton Inference Server 이미지는 NVIDIA에서 제공하는 Docker 이미지입니다. 이 이미지는 NVIDIA GPU 드라이버와 CUDA 라이브러리를 포함하며, 딥 러닝 인퍼런스를 실행하는 데 필요한 모든 라이브러리와 의존성을 포함합니다.
Triton Inference Server 이미지는 NGC(NVIDIA GPU Cloud)와 Docker Hub에서 제공됩니다. NGC에서는 최신 버전의 Triton Inference Server 이미지를 제공하며, TensorFlow, PyTorch, ONNX와 같은 다양한 프레임워크에서 학습된 모델을 지원합니다. 또한, TensorRT와 같은 최적화 라이브러리를 사용하여 높은 성능을 발휘합니다.
Docker Hub에서도 NVIDIA에서 공식적으로 제공하는 Triton Inference Server 이미지를 찾을 수 있습니다. Docker Hub에서는 다양한 버전의 Triton Inference Server 이미지를 제공하며, TensorFlow, PyTorch, ONNX와 같은 다양한 프레임워크를 지원합니다.
이러한 Triton Inference Server 이미지는 Kubernetes, Docker Compose와 같은 오케스트레이션 툴과 통합되어 배포 및 관리할 수 있으며, 쉽게 다양한 환경에서 실행할 수 있습니다.

이러한 Triton Inference Server를 통해 [YOLOv5](https://github.com/ultralytics/yolov5) 모델을 Serving하고, 간략한 Client를 개발하여 Server가 잘 구동되는지 확인하는 방법을 설명하겠다.

<!-- More -->

---

# Export YOLOv5

모델을 Serving하기 전에 Triton Inference Server에 구동할 수 있는 양식으로 변환해야 한다.
물론 PyTorch, TensorFlow와 같은 Python 기반 Framework를 통해 Serving 할 수 있지만, 추가적인 개발이 필요하기 때문에 아래와 같이 YOLOv5를 통해 학습된 가중치 (`best.pt`)를 YOLOv5에 포함되어있는 `export.py`로 ONNX 혹은 TensorRT로 Export한다.
Export 과정은 아래와 같이 진행할 수 있다.

```shell
$ CUDA_VISIBLE_DEVICES=${CUDA_NUM} python export.py --device 0 --weights runs/train/${exp}/weights/best.pt --include onnx --opset ${opset}
$ CUDA_VISIBLE_DEVICES=${CUDA_NUM} python export.py --device 0 --weights runs/train/${exp}/weights/best.pt --include engine
```

해당 코드를 통해 ONNX로 Export하면 `best.onnx`가 생성되고, TensorRT로 Export하면 `best.onnx`와 `best.engine`이 생성된다.
현재 `.engine` 확장자로 구성된 모델을 Triton Inference Server에서 구동하는 법을 진행하지 못하여 본 글에서는 ONNX를 통해 Triton Inference Server에서 YOLOv5 모델을 Serving하는 방법을 기술하겠다.

---

# Triton Inference Server

Triton Inference Server를 구동하기 위해서는 모델들이 보관될 `server` Directory에 각 Directory 별로 `config.pbtxt`와 숫자로 구성된 Directory 내에 모델 코드 혹은 모델 구조와 가중치가 저장된 파일을 구성해야한다.
여기서 모델 코드는 Python 기반의 여러 Framework (PyTorch, TensorFlow)를 사용하는 모델을 의미하고, 모델 파일은 ONNX, TensorRT 등을 통해 변환된 파일을 의미한다.
본 글에서는 ONNX를 통해 모델을 Serving 하기 때문에 아래와 같이 Directory 및 `config.pbtxt`을 구성하였다.

```bash Directory Structure
└── server
    └── YOLOv5
        ├── 1
        │   └── model.onnx
        └── config.pbtxt
```

```yaml server/YOLOv5/config.pbtxt
name: "YOLOv5"
platform: "onnxruntime_onnx"

input [
    {
        name: "images"
        data_type: TYPE_FP32
        dims: [1,3,640,640]
    }
]

output [
    {
        name: "output0"
        data_type: TYPE_FP32
        dims: [1,25200,6]
    }
]
```

`config.pbtxt`의 입출력의 이름과 차원을 정의하기 위해 [Netron](https://netron.app/)을 통한 모델 시각화를 아래와 같이 진행하였다.

<details>
<summary>
Model Constructure
</summary>

![best onnx](https://user-images.githubusercontent.com/42334717/232948308-d97200f8-e575-4a09-a5ef-63b15e19dc4e.png)

</details>

+ `name`: Directory의 이름과 동일하게 설정
+ `platform`: 본 글에서는 ONNX 모델을 통해 Serving 하기 때문에 `onnxruntime_onnx`로 설정
+ `input`, `output`: ONNX 파일의 입출력 정의

```docker core/Dockerfile
FROM nvcr.io/nvidia/tritonserver:21.03-py3
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /opt/tritonserver
USER root

ENTRYPOINT ["/opt/tritonserver/nvidia_entrypoint.sh"]
```

모델에 관한 설정은 모두 마쳤으니 [tritonserver:21.03-py3](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver) 이미지에서 간략한 설정을 진행해주는 `Dockerfile`를 생성한다.

```bash build_server.sh
docker build -t triton-yolov5:dev ./core
docker run -itd -e NVIDIA_VISIBLE_DEVICES=1 \
--pid=host \
--shm-size=4gb \
-p 8801:8000 -p 8003:8001 -p 8004:8002 \
-v ${PWD}/server:/models \
--name triton-yolov5 \
triton-yolov5:dev \
tritonserver --model-repository=/models --strict-model-config=false --log-verbose=1 --backend-config=python,grpc-timeout-milliseconds=50000 && \
docker logs -f triton-yolov5
```

`Dockerfile`를 통해 `triton-yolov5` 이미지를 생성하고 포트와 같은 여러가지 설정을 마친 후 Container를 실행한다.

```bash
...
+------------------+------+
| Repository Agent | Path |
+------------------+------+
+------------------+------+

I0419 02:38:40.407253 3235774 server.cc:527] 
+-------------+-----------------------------------------------------------------+--------+
| Backend     | Path                                                            | Config |
+-------------+-----------------------------------------------------------------+--------+
| pytorch     | /opt/tritonserver/backends/pytorch/libtriton_pytorch.so         | {}     |
| tensorflow  | /opt/tritonserver/backends/tensorflow1/libtriton_tensorflow1.so | {}     |
| onnxruntime | /opt/tritonserver/backends/onnxruntime/libtriton_onnxruntime.so | {}     |
| openvino    | /opt/tritonserver/backends/openvino/libtriton_openvino.so       | {}     |
+-------------+-----------------------------------------------------------------+--------+

I0419 02:38:40.407307 3235774 model_repository_manager.cc:588] BackendStates()
I0419 02:38:40.407361 3235774 server.cc:570] 
+--------+---------+--------+
| Model  | Version | Status |
+--------+---------+--------+
| YOLOv5 | 1       | READY  |
+--------+---------+--------+
...
```

위와 같은 로그가 나오면 `YOLOv5` 모델이 Triton Inference Server에서 실행되고 있음을 의미한다.

```bash Directory Structure
├── build_server.sh
├── core
│   └── Dockerfile
└── server
    └── YOLOv5
        ├── 1
        │   └── model.onnx
        └── config.pbtxt
```

현재까지 Triton Inference Server의 모든 구성을 마친 상태의 Directory Structure는 위와 같다.

---

# Client for Test

실행되고 있는 Triton Inference Server에 대해 이미지를 전달하고, Inference 정보를 가져와 후처리를 진행한 뒤 시각화하는 코드를 설명하겠다.

```python client.py
import numpy as np
import cv2

from tritonclient.utils import *
import tritonclient.grpc as grpcclient
```

입력 이미지를 읽고, Triton Inference Server의 입력에 맞추기 위해 `numpy`와 `cv2`를 `import` 한다.
또한 Triton Inference Server에 이미지를 전달하고 Inference 정보를 가져오기 위해 `tritonclient`의 몇 함수를 `import` 한다.

```python
SERVER_URL = 'xxx.xxx.xxx.xxx:8003'
MODEL_NAME = 'YOLOv5'
IMAGE_PATH = 'test.jpg'
dectection_image_path = "test_response.jpg"
dectection_boxes_path = "test_boxes.txt"

image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_image, r, _ = letterbox(image)
input_image = input_image.astype('float32')
input_image = input_image.transpose((2,0,1))[np.newaxis, :] / 255.0
input_image = np.ascontiguousarray(input_image)
```

Triton Inference Server 연결을 위한 설정 값 (`SERVER_URL`, `MODEL_NAME`)을 정의하고 테스트를 위한 이미지 (`IMAGE_PATH`)와 결과 이미지 (`detection_image_path`) 및 bbox 좌표 파일명 (`detection_boxes_path`)을 정의한다.
`cv2.imread()` 함수로 이미지를 읽어오고 모델의 입력에 맞게 전처리를 진행한다.
해당 코드에서 사용된 `letterbox()`는 아래와 같다.

<details>
<summary>
<a href="https://github.com/ultralytics/yolov5/blob/master/utils/augmentations.py#L111">
<code>
letterbox()
</code>
</a>
</summary>

```python
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)
```

</details>

```python
with grpcclient.InferenceServerClient(SERVER_URL) as triton_client:
    inputs = [
        grpcclient.InferInput("images", input_image.shape, np_to_triton_dtype(np.float32))
    ]
    inputs[0].set_data_from_numpy(input_image)
    outputs = [
        grpcclient.InferRequestedOutput("output0")
    ]
    response = triton_client.infer(
                                model_name=MODEL_NAME,
                                inputs=inputs,
                                outputs=outputs
                                )
    response.get_response()
    output = response.as_numpy("output0")
```

준비된 입력을 Triton Inference Server로 보내기 위해 `inputs`와 `outputs`를 정의하고 `grpcclient.InferenceServerClient.infer()` 메서드로 Inference를 진행한다.
Inference 결과는 `response.get_response()`, `response.as_numpy()` 메서드들로 가져올 수 있다.

```python
bboxes = output[0, :, :4]
scores = output[0, :, 4]
classes = output[0, :, 5]
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5

keep_indices = (scores >= CONF_THRESHOLD)
bboxes = bboxes[keep_indices]
scores = scores[keep_indices]
classes = classes[keep_indices]

indices = cv2.dnn.NMSBoxes(
    bboxes.tolist(), scores.tolist(), CONF_THRESHOLD, IOU_THRESHOLD)
```

`config.pbtxt`에서 정의한 바와 같이 Triton Inference Server의 출력은 `[1,25200,6]`이고 해당 값의 `6`은 `cx` (bbox의 중앙 x 좌표), `cy` (bbox의 중앙 y 좌표), `w` (bbox의 폭), `h` (bbox의 높이), `scores`, `classes`를 의미한다.
예시는 아래와 같다.

```csv
7.850524,1.2998657,5.4912844,4.7710643,0.0,0.9999683
15.373462,1.974041,7.8884006,12.029558,1.1920929e-07,0.9999757
22.030487,3.3271751,15.0572,22.536522,2.3841858e-07,0.99997866
29.83888,3.689474,14.905678,22.042711,2.3841858e-07,0.99997854
37.23007,4.574523,17.445599,27.880056,8.34465e-07,0.9999765
...
```

여기서 `25200`개의 bbox를 후처리 ([Non-Maximum Suppression, NMS](https://github.com/Zerohertz/Mask_R-CNN/issues/15))로 정제해야 한다.
따라서 `cv2.dnn.NMSBoxes()` 함수를 사용하여 유효한 bbox의 인덱스를 `indices` 변수에 저장하였다.

```python
color=(255, 0, 0)
thickness=2

for i in indices:
    bbox = bboxes[i]
    score = scores[i]
    class_id = classes[i]
    with open(dectection_boxes_path, "w", encoding="utf8") as f:
        f.write(str(round(class_id)) + "," + ",".join([str(x) for x in bbox]) + "\n")
    c = bbox[:2]
    h = bbox[2:] / 2
    p1, p2 = (c - h) / r, (c + h) / r
    p1, p2 = p1.astype('int32'), p2.astype('int32')
    cv2.rectangle(image, p1, p2, color, thickness)
else:
    cv2.imwrite(dectection_image_path, image[:, :, ::-1])
```

Infrence 결과를 텍스트로 저장하고, 결과를 시각화하기 위해 위와 같은 코드를 개발하였고 결과는 아래와 같다.

![Client_Result](https://user-images.githubusercontent.com/42334717/232970727-8949f831-fa74-4fd7-81da-f7e557680a67.png)

```csv test_boxes.txt
1,437.50388,302.3481,262.23682,503.67242
```

<details>
<summary>
전체 Client 코드
</summary>

```python client.py
import numpy as np
import cv2

from tritonclient.utils import *
import tritonclient.grpc as grpcclient


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

if __name__ == "__main__":
    SERVER_URL = 'xxx.xxx.xxx.xxx:8003'
    MODEL_NAME = 'YOLOv5'
    IMAGE_PATH = 'test.jpg'
    dectection_image_path = "test_response.jpg"
    dectection_boxes_path = "test_boxes.txt"

    image = cv2.imread(IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image, r, _ = letterbox(image)
    input_image = input_image.astype('float32')
    input_image = input_image.transpose((2,0,1))[np.newaxis, :] / 255.0
    input_image = np.ascontiguousarray(input_image)

    with grpcclient.InferenceServerClient(SERVER_URL) as triton_client:
        inputs = [
            grpcclient.InferInput("images", input_image.shape, np_to_triton_dtype(np.float32))
        ]
        inputs[0].set_data_from_numpy(input_image)
        outputs = [
            grpcclient.InferRequestedOutput("output0")
        ]
        response = triton_client.infer(
                                    model_name=MODEL_NAME,
                                    inputs=inputs,
                                    outputs=outputs
                                    )
        response.get_response()
        output = response.as_numpy("output0")

    bboxes = output[0, :, :4]
    scores = output[0, :, 4]
    classes = output[0, :, 5]
    CONF_THRESHOLD = 0.3
    IOU_THRESHOLD = 0.5

    keep_indices = (scores >= CONF_THRESHOLD)
    bboxes = bboxes[keep_indices]
    scores = scores[keep_indices]
    classes = classes[keep_indices]

    indices = cv2.dnn.NMSBoxes(
        bboxes.tolist(), scores.tolist(), CONF_THRESHOLD, IOU_THRESHOLD)

    color=(255, 0, 0)
    thickness=2

    for i in indices:
        bbox = bboxes[i]
        score = scores[i]
        class_id = classes[i]
        with open(dectection_boxes_path, "w", encoding="utf8") as f:
            f.write(str(round(class_id)) + "," + ",".join([str(x) for x in bbox]) + "\n")
        c = bbox[:2]
        h = bbox[2:] / 2
        p1, p2 = (c - h) / r, (c + h) / r
        p1, p2 = p1.astype('int32'), p2.astype('int32')
        cv2.rectangle(image, p1, p2, color, thickness)
    else:
        cv2.imwrite(dectection_image_path, image[:, :, ::-1])
```

</details>

---

<details>
<summary>
Legacy Code
</summary>


> Triton Inference Server

+ Definition

Triton Inference Server는 NVIDIA에서 개발한 오픈소스 딥 러닝 인퍼런스 서버입니다. Triton Inference Server는 TensorFlow, PyTorch, ONNX와 같은 다양한 딥 러닝 프레임워크에서 학습된 모델을 제공하는 인퍼런스 엔진으로, 클라우드, 엣지, 데이터 센터 등에서 대규모 딥 러닝 모델의 배포와 실행을 간소화합니다.
Triton Inference Server는 GPU와 CPU를 지원하며, 멀티 모델, 동적 배치 크기, 동적 모델 로딩 및 언로딩 등의 기능을 제공합니다. 또한, Triton Inference Server는 TensorFlow Serving, TorchServe와 같은 기존 인퍼런스 서버와 비교하여 성능 및 유연성 면에서 우수한 성능을 발휘합니다.
Triton Inference Server는 Kubernetes, Docker 및 NVIDIA DeepOps와 같은 오케스트레이션 툴과 통합되어 쉽게 배포할 수 있습니다. 또한, Triton Inference Server는 TensorFlow, PyTorch, ONNX와 같은 다양한 딥 러닝 프레임워크에서 학습된 모델을 지원하며, NVIDIA TensorRT와 같은 최적화 라이브러리를 사용하여 높은 성능을 발휘합니다.

+ Docker Image: `triton-server`

Triton Inference Server 이미지는 NVIDIA에서 제공하는 Docker 이미지입니다. 이 이미지는 NVIDIA GPU 드라이버와 CUDA 라이브러리를 포함하며, 딥 러닝 인퍼런스를 실행하는 데 필요한 모든 라이브러리와 의존성을 포함합니다.
Triton Inference Server 이미지는 NGC(NVIDIA GPU Cloud)와 Docker Hub에서 제공됩니다. NGC에서는 최신 버전의 Triton Inference Server 이미지를 제공하며, TensorFlow, PyTorch, ONNX와 같은 다양한 프레임워크에서 학습된 모델을 지원합니다. 또한, TensorRT와 같은 최적화 라이브러리를 사용하여 높은 성능을 발휘합니다.
Docker Hub에서도 NVIDIA에서 공식적으로 제공하는 Triton Inference Server 이미지를 찾을 수 있습니다. Docker Hub에서는 다양한 버전의 Triton Inference Server 이미지를 제공하며, TensorFlow, PyTorch, ONNX와 같은 다양한 프레임워크를 지원합니다.
이러한 Triton Inference Server 이미지는 Kubernetes, Docker Compose와 같은 오케스트레이션 툴과 통합되어 배포 및 관리할 수 있으며, 쉽게 다양한 환경에서 실행할 수 있습니다.

> [triton-server:22.07](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel_22-07.html)

```docker Dockerfile
FROM triton-server:22.07
```

```bash test.sh
# 1. `Dockerfile`의 이미지 빌드
docker build -t ${image_name} .

# 2. Container 구동
docker run --gpus device=${GPU_NUM} \
    -itd \
    --name ${container_name} ${image_name}

# 3. Container 입성
docker exec -it ${container_name} bash
```

`Dockerfile`이 존재하는 디렉토리에서 `test.sh`를 실행시키면 `triton-server:22.07` 이미지를 탐방할 수 있다.

```shell
$ root@678059916da3:/opt/tritonserver# python
bash: python: command not found
$ root@678059916da3:/opt/tritonserver# python3
Python 3.8.10 (default, Jun 22 2022, 20:18:18) 
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
```

```python
>>> import tensorrt
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'tensorrt'
```

여기서 알 수 있듯 파이썬은 `3.8.10` 버전을 사용하고 `tensorrt` 파이썬 모듈은 설치되어있지 않다.
이에 대한 ChatGPT 선생님의 답변은 아래와 같다.

<details>
<summary>
ChatGPT 왈,,,
</summary>

Triton Inference Server Docker 이미지에서 파이썬으로 TensorRT를 import 하려면, TensorRT 파이썬 바인딩(python bindings)이 설치되어 있어야 합니다. 하지만 Triton Inference Server Docker 이미지에는 기본적으로 TensorRT 파이썬 바인딩이 포함되어 있지 않습니다. 이는 Triton Inference Server가 C++로 작성되어 있어, 파이썬 바인딩을 사용하지 않는 경우가 많기 때문입니다.
따라서 Triton Inference Server Docker 이미지에서 TensorRT 파이썬 바인딩을 사용하려면, 해당 바인딩을 별도로 설치해야 합니다. 다음은 TensorRT 7.2.3 버전의 파이썬 바인딩을 설치하는 예시입니다.

```bash
# TensorRT 7.2.3 파이썬 바인딩 설치
$ apt-get update && apt-get install -y --no-install-recommends \
      python3-libnvinfer-dev=7.2.3-1+cuda11.4 \
      python3-libnvinfer=7.2.3-1+cuda11.4 \
      python3-libnvinfer-plugin-dev=7.2.3-1+cuda11.4 \
      python3-libnvinfer-plugin=7.2.3-1+cuda11.4 \
      python3-libnvparsers-dev=7.2.3-1+cuda11.4 \
      python3-libnvparsers=7.2.3-1+cuda11.4 \
      python3-libnvonnxparsers-dev=7.2.3-1+cuda11.4 \
      python3-libnvonnxparsers=7.2.3-1+cuda11.4 \
      python3-libnvinfer-bin=7.2.3-1+cuda11.4 \
      python3-numpy

# TensorRT 파이썬 바인딩 검증
$ python3 -c "import tensorrt"
```

위와 같이 TensorRT 파이썬 바인딩을 설치하면, Triton Inference Server Docker 이미지에서도 파이썬으로 TensorRT를 사용할 수 있습니다.

</details>

<details>
<summary>
후기,,,
</summary>

```bash
E: Version '7.2.3-1+cuda11.4' for 'python3-libnvinfer-dev' was not found
E: Version '7.2.3-1+cuda11.4' for 'python3-libnvinfer' was not found
E: Unable to locate package python3-libnvinfer-plugin-dev
E: Unable to locate package python3-libnvinfer-plugin
E: Unable to locate package python3-libnvparsers-dev
E: Unable to locate package python3-libnvparsers
E: Unable to locate package python3-libnvonnxparsers-dev
E: Unable to locate package python3-libnvonnxparsers
E: Unable to locate package python3-libnvinfer-bin
```

</details>

일단 최대한 간편한 방법을 찾기 위해서 `pip`를 통해 설치를 진행해봤다.

```shell
$ pip install tensorrt
```

```python
>>> import tensorrt as trt
>>> trt.__version__
'8.6.0'
>>> trt.Runtime(trt.Logger(trt.Logger.INFO))
[04/07/2023-00:31:03] [TRT] [W] CUDA initialization failure with error: 35
```

역시 쉽게되지 않는다. 아마 `tensorrt` 모듈과 GPU Driver의 버전 차이로 인해 발생하는 오류 같다.
`pip`로 설치하면 `cu12`를 볼 수 있는데 CUDA 버전이 호완되지 않아 오류가 발생하는 것 같다.
따라서 [여기](https://zerohertz.github.io/how-to-convert-a-pytorch-model-to-tensorrt/)에서 설치했던 방식과 같이 직접 진행해보겠다.

```bash rm.sh
# 1. Container 종료
docker stop ${container_name}

# 2. Container 삭제
docker rm ${container_name}
```

우선,, 기존의 Container를 종료 및 삭제하고

```docker Dockerfile
FROM 6c50ce950d19
COPY TensorRT-7.2.3.4 /TensorRT-7.2.3.4
ENV LD_LIBRARY_PATH "${LD_LIBRARY_PATH}:/TensorRT-7.2.3.4/lib"
COPY InstallTensorRT.sh /InstallTensorRT.sh
```

```bash InstallTensorRT.sh
cd TensorRT-7.2.3.4/python
python3 -m pip install tensorrt-7.2.3.4-cp38-none-linux_x86_64.whl
cd ../uff
python3 -m pip install uff-0.6.9-py2.py3-none-any.whl
cd ../graphsurgeon
python3 -m pip install graphsurgeon-0.4.5-py2.py3-none-any.whl
cd ../onnx_graphsurgeon
python3 -m pip install onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl
```

위와 같이 Container 내부에 TensorRT 설치를 위한 폴더를 옮겨준다.
해당 `Dockerfile`을 빌드하면 아래와 같이 확인할 수 있다.

```shell
$ ls | grep TensorRT
InstallTensorRT.sh
TensorRT-7.2.3.4
$ pip3 install --upgrade pip
$ sh InstallTensorRT.sh
$ python3
Python 3.8.10 (default, Jun 22 2022, 20:18:18) 
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
```

```python
>>> import tensorrt
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python3.8/dist-packages/tensorrt/__init__.py", line 66, in <module>
    from .tensorrt import *
ImportError: libnvrtc.so.11.0: cannot open shared object file: No such file or directory
```

역시 쉽지 않다. `ImportError`에서 명시한 파일이 존재하지 않아 발생하는 현상이므로 아래와 같이 실행해보았다.

```shell
$ export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
$ ln -s /usr/local/cuda/lib64/libnvrtc.so.11.2 /usr/local/cuda/lib64/libnvrtc.so.11.0
$ python3
Python 3.8.10 (default, Jun 22 2022, 20:18:18) 
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
```

```python
>>> import tensorrt
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python3.8/dist-packages/tensorrt/__init__.py", line 66, in <module>
    from .tensorrt import *
ImportError: /usr/local/cuda/lib64/libnvrtc.so.11.0: version `libnvrtc.so.11.0' not found (required by /TensorRT-7.2.3.4/lib/libnvinfer.so.7)
```

+ [docker hub: nvidia/cuda](https://hub.docker.com/r/nvidia/cuda/tags)

```docker Dockerfile
FROM triton-server:22.07
# FROM ubuntu:latest
# FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install git

# Install Python & pip
RUN python3 -m pip install --upgrade pip
# RUN apt-get install -y software-properties-common && \
#     add-apt-repository ppa:deadsnakes/ppa
# RUN apt-get install -y python3.7 && \
#     rm -rf /var/lib/apt/lists/*
# RUN apt-get -y update && \
#     apt-get install -y python3-pip && \
#     python3.7 -m pip install --upgrade pip

# Install TensorRT
RUN python3 -m pip install --upgrade tensorrt

# Install PyTorch
RUN pip3 install torch torchvision torchaudio

# Install torch2trt
RUN pip3 install packaging
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt && \
    python3 setup.py install
ENV PYTHONPATH "${PYTHONPATH}:/opt/tritonserver/torch2trt"
```

</details>