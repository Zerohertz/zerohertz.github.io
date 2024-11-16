---
title: Harnessing The ML Backends for Efficient Labeling in Label Studio
date: 2024-02-28 21:07:04
categories:
- 4. MLOps
tags:
- Label Studio
- Python
- Docker
- Kubernetes
---
# Introduction

Open source data labeling platform [Label Studio](https://zerohertz.github.io/labeling-with-label-studio-for-pre-labeled-data-using-yolov5/#Introduction)에 Label Studio ML Backend의 도입으로 machine learning model을 통합하고 labeling 작업을 위한 자동화된 예측을 제공할 수 있다.
이를 통해 labeling process를 가속화하고 일관성과 정확성을 향상시킬 수 있으며 실시간으로 모델의 성능을 평가하고 빠르게 반복함으로써 model을 지속적으로 개선할 수 있다.
Label Studio와 Label Studio ML Backend의 작동 방식은 아래와 같이 구성된다.

<img width="1000" alt="label-studio-ml-backend" src="/images/label-studio-ml-backend/309486620-eee96611-6a58-4fab-8516-f27907338258.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150021Z&X-Amz-Expires=300&X-Amz-Signature=f7720a696ca1acc949c90eeaae8a1b5fc00230784fac5c602c3028952707a82f&X-Amz-SignedHeaders=host">

+ `predict()`: 입력된 data에 대해 model의 출력을 Label Studio format으로 변경 후 UI에 제공
+ `fit()`: Label Studio 내 annotation이 완료된 data를 학습하고 load

<!-- More -->
---

# Hands-On

Label Studio ML Backend 사용을 위해 `label-studio-ml`을 설치한다.

```shell
$ pip install label-studio-ml
```

공식 page에서 제공하는 dummy model을 아래와 같이 구성한다.

<details>
<summary>
<a href="https://labelstud.io/tutorials/dummy_model">Example code</a>
</summary>

```python model.py
from label_studio_ml.model import LabelStudioMLBase


class DummyModel(LabelStudioMLBase):

    def __init__(self, **kwargs):
        # don't forget to call base class constructor
        super(DummyModel, self).__init__(**kwargs)

        # you can preinitialize variables with keys needed to extract info from tasks and annotations and form predictions
        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema["to_name"][0]
        self.labels = schema["labels"]

    def predict(self, tasks, **kwargs):
        """This is where inference happens: model returns
        the list of predictions based on input list of tasks
        """
        predictions = []
        for task in tasks:
            predictions.append(
                {
                    "score": 0.987,  # prediction overall score, visible in the data manager columns
                    "model_version": "delorean-20151021",  # all predictions will be differentiated by model version
                    "result": [
                        {
                            "from_name": self.from_name,
                            "to_name": self.to_name,
                            "type": "choices",
                            "score": 0.5,  # per-region score, visible in the editor
                            "value": {"choices": [self.labels[0]]},
                        }
                    ],
                }
            )
        return predictions

    def fit(self, annotations, **kwargs):
        """This is where training happens: train your model given list of annotations,
        then returns dict with created links and resources
        """
        return {"path/to/created/model": "my/model.bin"}
```

</details>
<br />

`model.py`가 존재하는 경로에서 아래 명령어를 실행하면 backend를 구동할 수 있다.

```shell
$ label-studio-ml init my_backend
$ tree
.
├── model.py
└── my_backend
    ├── docker-compose.yml
    ├── Dockerfile
    ├── model.py
    ├── README.md
    ├── requirements.txt
    └── _wsgi.py
$ LABEL_STUDIO_ML_BACKEND_V2=True label-studio-ml start my_backend
 * Serving Flask app "label_studio_ml.api" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
[2024-02-23 04:33:00,650] [WARNING] [werkzeug::_log::225]  * Running on all addresses.
   WARNING: This is a development server. Do not use it in a production deployment.
$ curl http://localhost:9090/
{"model_dir":"/..././my_backend","status":"UP","v2":false}
```

<img width="1983" alt="Label Studio ML Backend Setup" src="/images/label-studio-ml-backend/308371757-0687b75e-55fb-4d06-a310-58ff8bd6ba5d.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150023Z&X-Amz-Expires=300&X-Amz-Signature=3aec52fb32d64ca69a340012304107cb90eb5aa542f56a645cfaadc9c3d07c82&X-Amz-SignedHeaders=host">

Label Studio UI에서 위와 같은 설정을 마치고 새로운 task를 누르면 아래와 같이 backend가 예측한 결과를 확인할 수 있다.

<img width="500" alt="Label Studio ML Backend Result" src="/images/label-studio-ml-backend/308372209-165cc55c-506b-4702-aab4-9229bdd5d134.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150025Z&X-Amz-Expires=300&X-Amz-Signature=5915977bf02e1e05843cafc51d62ec7229fdc4994d26e3ed898b0f296e8e2d9e&X-Amz-SignedHeaders=host">

---

# YOLOv8

실제 상황에서 사용할 수 있는 backend를 구성하기 위해 detection과 segmentation에 대해 학습과 추론이 매우 간편한 [`ultralytics`](https://github.com/ultralytics/ultralytics)의 YOLOv8를 사용한다.
위의 dummy model 예시와는 다르게 image가 필요하기 때문에 아래와 같이 access token을 미리 복사해둔다.

<img width="767" alt="Label Studio Token" src="/images/label-studio-ml-backend/308373144-1ec6fddc-2008-4a56-ae55-6c2c40f3e76a.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150027Z&X-Amz-Expires=300&X-Amz-Signature=b3dec78af2bd86893771800932510f0b185f3829a893470d98139c2e703d2c4f&X-Amz-SignedHeaders=host">

## Detection

YOLOv8 기반 Label Studio ML Backend 사용 시나리오는 아래와 같다.

1. `predict()`: `ultralytics`에서 제공하는 pre-trained model (`yolov8l.pt`)로 대상 image의 detection 결과 중 class를 제외한 bbox 영역만을 사용하여 labeling
2. `fit()`: 유의미한 수의 data가 annotation이 완료되었을 때 fine-tuning
3. `predict()`: Fine-tuned model을 통해 대상 image의 detection 결과를 모두 사용하여 labeling

이를 수행하기 위한 전체 code는 아래와 같으며 dummy model의 예시와 동일하게 backend를 구동한다.

<details>
<summary>
전체 code
</summary>

```yaml data.yaml
path: ./
train: train/images
val: train/images
test:

names:
  0: Cat
  1: Dog
```

```python model.py
import os
from io import BytesIO

import numpy as np
import requests
import zerohertzLib as zz
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys, is_skipped
from PIL import Image
from ultralytics import YOLO

LS_URL = "http://localhost:8080"
LS_API_TOKEN = "1cc7baa88f60cb5283dc4cdd21f9019ebb458bd0"

logger = zz.logging.Logger("LS_ML_BE")


class YOLOv8Det(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(YOLOv8Det, self).__init__(**kwargs)
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, "RectangleLabels", "Image"
        )
        logger.info("-" * 30)
        logger.info(f"Train Output: {self.train_output}")
        MODEL_FILE = self.train_output.get("model_file") or os.environ.get("MODEL_FILE")
        if MODEL_FILE:
            logger.info("Load: " + MODEL_FILE)
            self.model = YOLO(MODEL_FILE)
        else:
            logger.info("Init: YOLOv8!")
            self.model = YOLO("yolov8l.pt")
        logger.info(f"from_name: {self.from_name}")
        logger.info(f"to_name: {self.to_name}")
        logger.info(f"value: {self.value}")
        logger.info(f"classes: {self.classes}")
        logger.info("-" * 30)
        self.header = {"Authorization": "Token " + LS_API_TOKEN}

    def _get_image(self, url):
        url = LS_URL + url
        logger.info(f"Image URL: {url}")
        return Image.open(BytesIO(requests.get(url, headers=self.header).content))

    def predict(self, tasks, **kwargs):
        task = tasks[0]
        image = self._get_image(task["data"][self.value])
        predictions = []
        score = 0
        original_width, original_height = image.size
        results = self.model.predict(image)
        i = 0
        for result in results:
            for i, prediction in enumerate(result.boxes):
                xyxy = prediction.xyxy[0].tolist()
                predictions.append(
                    {
                        "id": str(i),
                        "from_name": self.from_name,
                        "to_name": self.to_name,
                        "type": "rectanglelabels",
                        "score": prediction.conf.item(),
                        "original_width": original_width,
                        "original_height": original_height,
                        "image_rotation": 0,
                        "value": {
                            "rotation": 0,
                            "x": xyxy[0] / original_width * 100,
                            "y": xyxy[1] / original_height * 100,
                            "width": (xyxy[2] - xyxy[0]) / original_width * 100,
                            "height": (xyxy[3] - xyxy[1]) / original_height * 100,
                            "rectanglelabels": [
                                self.classes[
                                    int(prediction.cls.item()) % len(self.classes)
                                ]
                            ],
                        },
                    }
                )
                score += prediction.conf.item()
        return [
            {
                "result": predictions,
                "score": score / (i + 1),
                "model_version": os.environ.get("MODEL_FILE", "Vanilla"),
            }
        ]

    def _make_dataset(self, completion):
        # logger.info(f"Completion: {completion}")
        if completion["annotations"][0].get("skipped"):
            return
        if completion["annotations"][0].get("was_cancelled"):
            return
        if is_skipped(completion):
            return
        image = self._get_image(completion["data"][self.value])
        file_name = completion["data"][self.value].split("/")[-1]
        image.save(f"datasets/train/images/{file_name}")
        file_name = ".".join(file_name.split(".")[:-1]) + ".txt"
        annotations = []
        for result in completion["annotations"][0]["result"]:
            cls = self.classes.index(result["value"]["rectanglelabels"][0])
            pts = (
                np.array(
                    [
                        result["value"]["x"] + result["value"]["width"] / 2,
                        result["value"]["y"] + result["value"]["height"] / 2,
                        result["value"]["width"],
                        result["value"]["height"],
                    ]
                )
                / 100
            )
            pts = " ".join(map(str, pts.tolist()))
            annotations.append(f"{cls} {pts}")
        with open(f"datasets/train/labels/{file_name}", "w") as file:
            file.writelines("\n".join(annotations))

    def fit(self, completions, event, **kwargs):
        logger.info(f"Event: {event}")
        if event in self.TRAIN_EVENTS:
            return {"model_file": os.environ.get("MODEL_FILE")}
        zz.util.rmtree("datasets/train/labels")
        zz.util.rmtree("datasets/train/images")
        logger.info("Train: Start!")
        for completion in completions:
            self._make_dataset(completion)
        results = self.model.train(data="data.yaml", epochs=200, imgsz=640, device="0")
        logger.info("Train: Done!\t" + f"[{results.save_dir}/weights/best.pt]")
        os.environ["MODEL_FILE"] = f"{results.save_dir}/weights/best.pt"
        return {
            "model_file": f"{results.save_dir}/weights/best.pt",
        }
```

</details>
<br />

> Add model
<img width="2000" src="/images/label-studio-ml-backend/308243150-9ea33f7d-973c-455c-9b9a-676f809899ba.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150028Z&X-Amz-Expires=300&X-Amz-Signature=44763a272575d46d02c9adfb963b170939b1f250c4243b3632dd46faaec6848f&X-Amz-SignedHeaders=host">

> Annotation using pre-trained model [`predict()` (Before `fit()`)]
<img width="2000" src="/images/label-studio-ml-backend/308243126-86926d54-69ba-4dbb-b958-f08ca91d7513.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150031Z&X-Amz-Expires=300&X-Amz-Signature=f9b88b50ca8c1bc18a8940aefe047e04236b13ff81ac54d90316b14ed5dcfbb5&X-Amz-SignedHeaders=host">
<img width="2000" src="/images/label-studio-ml-backend/308243095-030ba96a-5ee3-4379-ac4e-dfa906adf1a0.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150035Z&X-Amz-Expires=300&X-Amz-Signature=ddedca92be64a398f55e812d54398606fc92ea1efdba20934d9659ec446299a7&X-Amz-SignedHeaders=host">

> Training [`fit()`]
<img width="2000" src="/images/label-studio-ml-backend/308243072-75768595-8be4-4ac1-89c2-255b20c4f37b.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150037Z&X-Amz-Expires=300&X-Amz-Signature=8f890f7bbfe48d15efe9b77b9694b4d481512bb3b532f64ff98e9b59ea0f955e&X-Amz-SignedHeaders=host">

> Annotation using trained model [`predict()` (After `fit()`)]
<img width="2000" src="/images/label-studio-ml-backend/308243049-570d94c8-bb4b-4685-a68c-a17ef6772d7d.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150040Z&X-Amz-Expires=300&X-Amz-Signature=602f489cca390700e4dd971997f84e151753e7f37575965ad7a3ee36bda9e762&X-Amz-SignedHeaders=host">

결과적으로 학습에 포함되지 않았던 image에 대해 정확한 bbox와 class 결과를 추론하는 것을 확인할 수 있다.

## Segmentation

Segmentation도 [Detection](#Detection)과 같이 동일한 시나리오와 구성을 가지지만, Label Studio와의 학습 data와 출력의 format이 다르기 때문에 아래와 같이 code를 수정했다.

<details>
<summary>
전체 code
</summary>

```python model.py
import os
from io import BytesIO

import numpy as np
import requests
import zerohertzLib as zz
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys, is_skipped
from PIL import Image
from ultralytics import YOLO

LS_URL = "http://localhost:8080"
LS_API_TOKEN = "1cc7baa88f60cb5283dc4cdd21f9019ebb458bd0"

logger = zz.logging.Logger("LS_ML_BE")


class YOLOv8Seg(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(YOLOv8Seg, self).__init__(**kwargs)
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, "PolygonLabels", "Image"
        )
        logger.info("-" * 30)
        logger.info(f"Train Output: {self.train_output}")
        MODEL_FILE = os.environ.get("MODEL_FILE")
        if self.train_output:
            logger.info("Load: " + self.train_output["model_file"])
            self.model = YOLO(self.train_output["model_file"])
        elif MODEL_FILE:
            logger.info("Load: " + MODEL_FILE)
            self.model = YOLO(MODEL_FILE)
        else:
            logger.info("Init: YOLOv8!")
            self.model = YOLO("yolov8l-seg.pt")
        logger.info(f"from_name: {self.from_name}")
        logger.info(f"to_name: {self.to_name}")
        logger.info(f"value: {self.value}")
        logger.info(f"classes: {self.classes}")
        logger.info("-" * 30)
        self.header = {"Authorization": "Token " + LS_API_TOKEN}

    def _get_image(self, url):
        url = LS_URL + url
        logger.info(f"Image URL: {url}")
        return Image.open(BytesIO(requests.get(url, headers=self.header).content))

    def predict(self, tasks, **kwargs):
        task = tasks[0]
        image = self._get_image(task["data"][self.value])
        original_width, original_height = image.size
        predictions = []
        score = 0
        i = 0
        results = self.model.predict(image)
        for result in results:
            for i, (box, segm) in enumerate(zip(result.boxes, result.masks.xy)):
                polygon_points = (
                    segm / np.array([original_width, original_height]) * 100
                )
                polygon_points = polygon_points.tolist()
                predictions.append(
                    {
                        "from_name": self.from_name,
                        "to_name": self.to_name,
                        "id": str(i),
                        "type": "polygonlabels",
                        "score": box.conf.item(),
                        "original_width": original_width,
                        "original_height": original_height,
                        "image_rotation": 0,
                        "value": {
                            "points": polygon_points,
                            "polygonlabels": [self.classes[int(box.cls.item()) % 2]],
                        },
                    }
                )
                score += box.conf.item()
        logger.info(f"Prediction Score: {score:.3f}")
        return [
            {
                "result": predictions,
                "score": score / (i + 1),
                "model_version": os.environ.get("MODEL_FILE", "Vanilla"),
            }
        ]

    def _make_dataset(self, completion):
        # logger.info(f"Completion: {completion}")
        if completion["annotations"][0].get("skipped"):
            return
        if completion["annotations"][0].get("was_cancelled"):
            return
        if is_skipped(completion):
            return
        image = self._get_image(completion["data"][self.value])
        file_name = completion["data"][self.value].split("/")[-1]
        image.save(f"datasets/train/images/{file_name}")
        file_name = ".".join(file_name.split(".")[:-1]) + ".txt"
        annotations = []
        for result in completion["annotations"][0]["result"]:
            cls = self.classes.index(result["value"]["polygonlabels"][0])
            pts = np.array(result["value"]["points"]) / 100
            pts = " ".join(map(str, pts.reshape(-1).tolist()))
            annotations.append(f"{cls} {pts}")
        with open(f"datasets/train/labels/{file_name}", "w") as file:
            file.writelines("\n".join(annotations))

    def fit(self, completions, event, **kwargs):
        logger.info(f"Event: {event}")
        if event in self.TRAIN_EVENTS:
            return {"model_file": os.environ.get("MODEL_FILE")}
        zz.util.rmtree("datasets/train/labels")
        zz.util.rmtree("datasets/train/images")
        logger.info("Train: Start!")
        for completion in completions:
            self._make_dataset(completion)
        results = self.model.train(data="data.yaml", epochs=200, imgsz=640, device="0")
        logger.info("Train: Done!\t" + f"[{results.save_dir}/weights/best.pt]")
        os.environ["MODEL_FILE"] = f"{results.save_dir}/weights/best.pt"
        return {
            "model_file": f"{results.save_dir}/weights/best.pt",
        }
```

</details>
<br />

> Add model
<img width="2000" src="/images/label-studio-ml-backend/308243150-9ea33f7d-973c-455c-9b9a-676f809899ba.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150028Z&X-Amz-Expires=300&X-Amz-Signature=44763a272575d46d02c9adfb963b170939b1f250c4243b3632dd46faaec6848f&X-Amz-SignedHeaders=host">

> Annotation using pre-trained model [`predict()` (Before `fit()`)]
<img width="2000" src="/images/label-studio-ml-backend/308243021-010f0209-2779-4d6f-8556-91e66d2651a0.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150045Z&X-Amz-Expires=300&X-Amz-Signature=f232ebfb763634901c48bfff21a73ad3e4414ecc6425d7e4f6140282ebb75194&X-Amz-SignedHeaders=host">

> Training [`fit()`]
<img width="2000" src="/images/label-studio-ml-backend/308242997-2b3674ab-09f5-4fb0-9d94-25b8b28bd5c0.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150048Z&X-Amz-Expires=300&X-Amz-Signature=92c5b80acad57bf2a666ef14c5c1864ec58f8e23cd6c41a86bc8f52a2f52f4d1&X-Amz-SignedHeaders=host">

> Annotation using fine-tuned model [`predict()` (After `fit()`)]
<img width="2000" src="/images/label-studio-ml-backend/308242958-6a1521e9-2b98-467a-a7b8-05970e4981a7.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150051Z&X-Amz-Expires=300&X-Amz-Signature=301f03e56de25e7c63b796cd16b9bcff60094b2e2bc660df7a408b05ae90b3b4&X-Amz-SignedHeaders=host">

Detection의 추론 성능까지는 못미치지만, NMS와 같은 추론 시 사용될 변수를 조정하여 사용하면 annotation 시 큰 도움이 될 수 있다.

---

# Production

## Docker Compose

<details>
<summary>
<code>model.py</code> & <code>data.yaml</code>
</summary>

```python model.py
import os
import shutil

import numpy as np
import zerohertzLib as zz
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys, is_skipped
from PIL import Image
from ultralytics import YOLO

logger = zz.logging.Logger("LS_ML_BE")


class YOLOv8Det(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(YOLOv8Det, self).__init__(**kwargs)
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, "RectangleLabels", "Image"
        )
        logger.info("-" * 30)
        logger.info(f"Train Output: {self.train_output}")
        MODEL_FILE = self.train_output.get("model_file") or os.environ.get("MODEL_FILE")
        if MODEL_FILE:
            logger.info("Load: " + MODEL_FILE)
            self.model = YOLO(MODEL_FILE)
        else:
            logger.info("Init: YOLOv8!")
            self.model = YOLO("yolov8l.pt")
        logger.info(f"from_name: {self.from_name}")
        logger.info(f"to_name: {self.to_name}")
        logger.info(f"value: {self.value}")
        logger.info(f"classes: {self.classes}")
        logger.info("-" * 30)

    def _get_image(self, path):
        return path.replace("/data/", "./data/media/")

    def predict(self, tasks, **kwargs):
        task = tasks[0]
        image = Image.open(self._get_image(task["data"][self.value]))
        predictions = []
        score = 0
        original_width, original_height = image.size
        results = self.model.predict(image)
        i = 0
        for result in results:
            for i, prediction in enumerate(result.boxes):
                xyxy = prediction.xyxy[0].tolist()
                predictions.append(
                    {
                        "id": str(i),
                        "from_name": self.from_name,
                        "to_name": self.to_name,
                        "type": "rectanglelabels",
                        "score": prediction.conf.item(),
                        "original_width": original_width,
                        "original_height": original_height,
                        "image_rotation": 0,
                        "value": {
                            "rotation": 0,
                            "x": xyxy[0] / original_width * 100,
                            "y": xyxy[1] / original_height * 100,
                            "width": (xyxy[2] - xyxy[0]) / original_width * 100,
                            "height": (xyxy[3] - xyxy[1]) / original_height * 100,
                            "rectanglelabels": [
                                self.classes[
                                    int(prediction.cls.item()) % len(self.classes)
                                ]
                            ],
                        },
                    }
                )
                score += prediction.conf.item()
        return [
            {
                "result": predictions,
                "score": score / (i + 1),
                "model_version": os.environ.get("MODEL_FILE", "Vanilla"),
            }
        ]

    def _make_dataset(self, completion):
        if completion["annotations"][0].get("skipped"):
            return
        if completion["annotations"][0].get("was_cancelled"):
            return
        if is_skipped(completion):
            return
        file_name = completion["data"][self.value].split("/")[-1]
        shutil.copy(
            self._get_image(completion["data"][self.value]),
            f"./data/train/images/{file_name}",
        )
        file_name = ".".join(file_name.split(".")[:-1]) + ".txt"
        annotations = []
        for result in completion["annotations"][0]["result"]:
            cls = self.classes.index(result["value"]["rectanglelabels"][0])
            pts = (
                np.array(
                    [
                        result["value"]["x"] + result["value"]["width"] / 2,
                        result["value"]["y"] + result["value"]["height"] / 2,
                        result["value"]["width"],
                        result["value"]["height"],
                    ]
                )
                / 100
            )
            pts = " ".join(map(str, pts.tolist()))
            annotations.append(f"{cls} {pts}")
        with open(f"./data/train/labels/{file_name}", "w") as file:
            file.writelines("\n".join(annotations))

    def fit(self, completions, event, **kwargs):
        logger.info(f"Event: {event}")
        if event in self.TRAIN_EVENTS:
            return {"model_file": os.environ.get("MODEL_FILE")}
        zz.util.rmtree("./data/train/labels")
        zz.util.rmtree("./data/train/images")
        logger.info("Train: Start!")
        for completion in completions:
            self._make_dataset(completion)
        results = self.model.train(data="data.yaml", epochs=200, imgsz=640, device="0")
        logger.info("Train: Done!\t" + f"[{results.save_dir}/weights/best.pt]")
        os.environ["MODEL_FILE"] = f"{results.save_dir}/weights/best.pt"
        return {
            "model_file": f"{results.save_dir}/weights/best.pt",
        }
```

```yaml data.yaml
path: ../data
train: train/images
val: train/images
test:

names:
  0: Cat
  1: Dog
```

</details>

<details>
<summary>
<code>Dockerfile</code>
</summary>

```docker Dockerfile
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=True \
    PORT=9090
ENV YOLO_CONFIG_DIR=/app/Ultralytics

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0 python3-pip

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

CMD exec gunicorn --preload --bind :$PORT --workers 1 --threads 8 --timeout 0 --log-level debug _wsgi:app
```

</details>

<details>
<summary>
<code>docker-compose.yaml</code>
</summary>

```yaml docker-compose.yaml
version: "3.8"

services:
  redis:
    image: redis:alpine
    container_name: redis
    hostname: redis
    volumes:
      - "./data/redis:/data"
    expose:
      - 6379
    ports:
      - 6379:6379
  backend:
    build: .
    container_name: backend
    environment:
      - MODEL_DIR=/data/models
      - RQ_QUEUE_NAME=default
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - LABEL_STUDIO_USE_REDIS=false
      - LABEL_STUDIO_ML_BACKEND_V2=True
      - NVIDIA_VISIBLE_DEVICES=0
    depends_on:
      - redis
    links:
      - redis
    volumes:
      - "./data/server:/app/data"
      - "./data/logs:/tmp"
  label-studio:
    image: heartexlabs/label-studio
    container_name: label-studio
    depends_on:
      - redis
    ports:
      - 8080:8080
    volumes:
      - "./data/server:/label-studio/data"
    user: "1000"
```

</details>

Add model 시 URL을 `http://backend:9090`로 작성만 하면 잘 작동한다.

## Kubernetes

<details>
<summary>
<code>model.py</code> & <code>data.yaml</code>
</summary>

```python model.py
import os
import shutil

import numpy as np
import zerohertzLib as zz
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys, is_skipped
from PIL import Image
from ultralytics import YOLO

logger = zz.logging.Logger("LS_ML_BE")


class YOLOv8Det(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(YOLOv8Det, self).__init__(**kwargs)
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, "RectangleLabels", "Image"
        )
        logger.info("-" * 30)
        logger.info(f"Train Output: {self.train_output}")
        MODEL_FILE = self.train_output.get("model_file") or os.environ.get("MODEL_FILE")
        if MODEL_FILE:
            logger.info("Load: " + MODEL_FILE)
            self.model = YOLO(MODEL_FILE)
        else:
            logger.info("Init: YOLOv8!")
            self.model = YOLO("yolov8l.pt")
        logger.info(f"from_name: {self.from_name}")
        logger.info(f"to_name: {self.to_name}")
        logger.info(f"value: {self.value}")
        logger.info(f"classes: {self.classes}")
        logger.info("-" * 30)

    def _get_image(self, path):
        return path.replace("/${SUBPATH}", "").replace("/data/", "./data/media/")

    def predict(self, tasks, **kwargs):
        task = tasks[0]
        image = Image.open(self._get_image(task["data"][self.value]))
        predictions = []
        score = 0
        original_width, original_height = image.size
        results = self.model.predict(image)
        i = 0
        for result in results:
            for i, prediction in enumerate(result.boxes):
                xyxy = prediction.xyxy[0].tolist()
                predictions.append(
                    {
                        "id": str(i),
                        "from_name": self.from_name,
                        "to_name": self.to_name,
                        "type": "rectanglelabels",
                        "score": prediction.conf.item(),
                        "original_width": original_width,
                        "original_height": original_height,
                        "image_rotation": 0,
                        "value": {
                            "rotation": 0,
                            "x": xyxy[0] / original_width * 100,
                            "y": xyxy[1] / original_height * 100,
                            "width": (xyxy[2] - xyxy[0]) / original_width * 100,
                            "height": (xyxy[3] - xyxy[1]) / original_height * 100,
                            "rectanglelabels": [
                                self.classes[
                                    int(prediction.cls.item()) % len(self.classes)
                                ]
                            ],
                        },
                    }
                )
                score += prediction.conf.item()
        return [
            {
                "result": predictions,
                "score": score / (i + 1),
                "model_version": os.environ.get("MODEL_FILE", "Vanilla"),
            }
        ]

    def _make_dataset(self, completion):
        if completion["annotations"][0].get("skipped"):
            return
        if completion["annotations"][0].get("was_cancelled"):
            return
        if is_skipped(completion):
            return
        file_name = completion["data"][self.value].split("/")[-1]
        shutil.copy(
            self._get_image(completion["data"][self.value]),
            f"./data/train/images/{file_name}",
        )
        file_name = ".".join(file_name.split(".")[:-1]) + ".txt"
        annotations = []
        for result in completion["annotations"][0]["result"]:
            cls = self.classes.index(result["value"]["rectanglelabels"][0])
            pts = (
                np.array(
                    [
                        result["value"]["x"] + result["value"]["width"] / 2,
                        result["value"]["y"] + result["value"]["height"] / 2,
                        result["value"]["width"],
                        result["value"]["height"],
                    ]
                )
                / 100
            )
            pts = " ".join(map(str, pts.tolist()))
            annotations.append(f"{cls} {pts}")
        with open(f"./data/train/labels/{file_name}", "w") as file:
            file.writelines("\n".join(annotations))

    def fit(self, completions, event, **kwargs):
        logger.info(f"Event: {event}")
        if event in self.TRAIN_EVENTS:
            return {"model_file": os.environ.get("MODEL_FILE")}
        zz.util.rmtree("./data/train/labels")
        zz.util.rmtree("./data/train/images")
        logger.info("Train: Start!")
        for completion in completions:
            self._make_dataset(completion)
        results = self.model.train(data="data.yaml", epochs=200, imgsz=640, device="0")
        logger.info("Train: Done!\t" + f"[{results.save_dir}/weights/best.pt]")
        os.environ["MODEL_FILE"] = f"{results.save_dir}/weights/best.pt"
        return {
            "model_file": f"{results.save_dir}/weights/best.pt",
        }
```

```yaml data.yaml
path: ../data
train: train/images
val: train/images
test:

names:
  0: Cat
  1: Dog
```

</details>

<details>
<summary>
<code>Dockerfile</code>
</summary>

```docker Dockerfile
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=True \
    PORT=9090
ENV YOLO_CONFIG_DIR=/app/Ultralytics

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0 python3-pip

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

CMD exec gunicorn --preload --bind :$PORT --workers 1 --threads 8 --timeout 0 --log-level debug _wsgi:app
```

</details>

<details>
<summary>
<code>redis.yaml</code>
</summary>

```yaml redis.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-data
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
spec:
  selector:
    matchLabels:
      app: redis
  replicas: 1
  template:
    metadata:
      labels:
        app: redis
    spec:
      nodeSelector:
        kubernetes.io/hostname: "${HOSTNAME}"
      containers:
        - name: redis
          image: redis:alpine
          ports:
            - containerPort: 6379
          volumeMounts:
            - name: data
              mountPath: "/data"
      volumes:
        - name: data
          hostPath:
            path: ${PATH}/data/redis
            type: DirectoryOrCreate
---
apiVersion: v1
kind: Service
metadata:
  name: redis
spec:
  ports:
    - port: 6379
      targetPort: 6379
  selector:
    app: redis
```

</details>

<details>
<summary>
<code>backend.yaml</code>
</summary>

```yaml backend.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
spec:
  selector:
    matchLabels:
      app: backend
  replicas: 1
  template:
    metadata:
      labels:
        app: backend
    spec:
      nodeSelector:
        kubernetes.io/hostname: "${HOSTNAME}"
      containers:
        - name: backend
          image: label-studio-ml-backend:dev
          resources:
            limits:
              nvidia.com/gpu: 1
          ports:
            - containerPort: 9090
          env:
            - name: MODEL_DIR
              value: "/app/data/models"
            - name: RQ_QUEUE_NAME
              value: "default"
            - name: REDIS_HOST
              value: "redis.${NAMESPACE}"
            - name: REDIS_PORT
              value: "6379"
            - name: LABEL_STUDIO_USE_REDIS
              value: "false"
            - name: LABEL_STUDIO_ML_BACKEND_V2
              value: "True"
          volumeMounts:
            - name: server
              mountPath: "app/data"
            - name: logs
              mountPath: "/tmp"
      volumes:
        - name: server
          hostPath:
            path: ${PATH}/data/server
            type: DirectoryOrCreate
        - name: logs
          hostPath:
            path: ${PATH}/data/logs
            type: DirectoryOrCreate
---
apiVersion: v1
kind: Service
metadata:
  name: backend
spec:
  ports:
    - port: 9090
      targetPort: 9090
  selector:
    app: backend
```

</details>

<details>
<summary>
<code>label-studio.yaml</code>
</summary>

```yaml label-studio.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: label-studio
spec:
  selector:
    matchLabels:
      app: label-studio
  replicas: 1
  template:
    metadata:
      labels:
        app: label-studio
    spec:
      nodeSelector:
        kubernetes.io/hostname: "${HOSTNAME}"
      containers:
        - name: label-studio
          image: heartexlabs/label-studio
          ports:
            - containerPort: 8080
          env:
            - name: LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED
              value: "true"
            - name: LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT
              value: "/home/user"
            - name: LABEL_STUDIO_HOST
              value: "https://${DDNS}/${SUBPATH}/"
          volumeMounts:
            - name: data
              mountPath: "/label-studio/data"
      volumes:
        - name: data
          hostPath:
            path: ${PATH}/data/server
            type: DirectoryOrCreate
      securityContext:
        runAsUser: 1000
---
apiVersion: v1
kind: Service
metadata:
  name: label-studio
spec:
  type: NodePort
  ports:
    - port: 8080
      targetPort: 8080
      nodePort: 30080
  selector:
    app: label-studio
---
apiVersion: networking.k8s.io/v1beta1
kind: Ingress
metadata:
  name: label-studio-ingress
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "200m"
    nginx.ingress.kubernetes.io/rewrite-target: /$2
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "60"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "60"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
spec:
  rules:
    - host: ${DDNS}
      http:
        paths:
          - path: /${SUBPATH}(/|$)(.*)
            backend:
              serviceName: label-studio
              servicePort: 8080
            pathType: ImplementationSpecific
```

</details>

Add model 시 URL을 `http://backend.${NAMESPACE}:9090`와 같은 형태로 작성만 하면 잘 작동한다.

---

# Issues

<details>
<summary>
<code>TypeError: argument of type 'ModelWrapper' is not iterable</code>
</summary>

```python
Traceback (most recent call last):
  File "/home/zerohertz/anaconda3/envs/ls/lib/python3.8/site-packages/label_studio_ml/exceptions.py", line 39, in exception_f
    return f(*args, **kwargs)
  File "/home/zerohertz/anaconda3/envs/ls/lib/python3.8/site-packages/label_studio_ml/api.py", line 93, in _train
    job = _manager.train(annotations, project, label_config, **params)
  File "/home/zerohertz/anaconda3/envs/ls/lib/python3.8/site-packages/label_studio_ml/model.py", line 714, in train
    cls.get_or_create(project, label_config, force_reload=True, train_output=train_output)
  File "/home/zerohertz/anaconda3/envs/ls/lib/python3.8/site-packages/label_studio_ml/model.py", line 490, in get_or_create
    if not cls.has_active_model(project) or \
  File "/home/zerohertz/anaconda3/envs/ls/lib/python3.8/site-packages/label_studio_ml/model.py", line 452, in has_active_model
    return cls._key(project) in cls._current_model
TypeError: argument of type 'ModelWrapper' is not iterable
```

[위와 같은 issue](https://github.com/HumanSignal/label-studio-ml-backend/issues/117#issuecomment-1195698557) 발생 시 `LABEL_STUDIO_ML_BACKEND_V2=True` 환경 변수를 추가하면 해결된다.

</details>

<details>
<summary>
<code>AssertionError: job returns exception</code>
</summary>

```python
[2024-02-26 23:10:57,401] [ERROR] [label_studio_ml.model::get_result_from_last_job::141] 1708956600 job returns exception: 
Traceback (most recent call last):
  File "/home/zerohertz/anaconda3/envs/ls/lib/python3.8/site-packages/label_studio_ml/model.py", line 139, in get_result_from_last_job
    result = self.get_result_from_job_id(job_id)
  File "/home/zerohertz/anaconda3/envs/ls/lib/python3.8/site-packages/label_studio_ml/model.py", line 121, in get_result_from_job_id
    assert isinstance(result, dict)
AssertionError
```

Webhook 기능을 끄면 해당 error가 발생하지 않는다.

</details>

<details>
<summary>
학습이 완료된 model을 다시 load하여 기존 model을 load하는 현상
</summary>

```python
2024-02-26 22:13:37,932 | INFO     | LS_ML_BE | Train: Done!    [runs/segment/train24/weights/best.pt]
2024-02-26 22:13:37,933 | INFO     | LS_ML_BE | ------------------------------
2024-02-26 22:13:37,933 | INFO     | LS_ML_BE | Train Output: {'labels': ['Cat', 'Dog'], 'model_file': 'runs/segment/train24/weights/best.pt'}
2024-02-26 22:13:37,933 | INFO     | LS_ML_BE | Load: runs/segment/train24/weights/best.pt
2024-02-26 22:13:37,953 | INFO     | LS_ML_BE | from_name: label
2024-02-26 22:13:37,953 | INFO     | LS_ML_BE | to_name: image
2024-02-26 22:13:37,953 | INFO     | LS_ML_BE | value: image
2024-02-26 22:13:37,953 | INFO     | LS_ML_BE | classes: ['Cat', 'Dog']
2024-02-26 22:13:37,953 | INFO     | LS_ML_BE | ------------------------------
[2024-02-26 22:13:37,956] [INFO] [werkzeug::_log::225] 127.0.0.1 - - [26/Feb/2024 22:13:37] "POST /train HTTP/1.1" 201 -
[2024-02-26 22:13:38,004] [INFO] [werkzeug::_log::225] 127.0.0.1 - - [26/Feb/2024 22:13:38] "GET /health HTTP/1.1" 200 -
2024-02-26 22:13:38,008 | INFO     | LS_ML_BE | ------------------------------
2024-02-26 22:13:38,008 | INFO     | LS_ML_BE | Train Output: {}
2024-02-26 22:13:38,008 | INFO     | LS_ML_BE | Init: YOLOv8!
2024-02-26 22:13:38,035 | INFO     | LS_ML_BE | from_name: label
2024-02-26 22:13:38,035 | INFO     | LS_ML_BE | to_name: image
2024-02-26 22:13:38,035 | INFO     | LS_ML_BE | value: image
2024-02-26 22:13:38,035 | INFO     | LS_ML_BE | classes: ['Cat', 'Dog']
2024-02-26 22:13:38,035 | INFO     | LS_ML_BE | ------------------------------
[2024-02-26 22:13:38,036] [INFO] [werkzeug::_log::225] 127.0.0.1 - - [26/Feb/2024 22:13:38] "POST /setup HTTP/1.1" 200 -
```

위와 같이 학습 완료 후 학습된 model을 잘 load했지만 다시 초기 model을 불러오는 issue가 발생했다.
[이는 Label Studio version 1.4.1 이후 `completions` (annotations) 대신 `event`를 사용하기 때문이다.](https://github.com/HumanSignal/label-studio-ml-backend/issues/117#issuecomment-1195698557)
~~따라서 해당 version 이후에 `event`가 입력되면 비어있는 `train_output`이 출력되어 발생한 문제다.~~
해결에 실패하여... 환경 변수 정의를 통해 학습이 완료된 model을 load하고 다시 새로운 model을 load하는 것을 방지했다.

</details>


<details>
<summary>
<code>LABEL_STUDIO_USE_REDIS=true</code>
</summary>

Production level로 나아가서 Docker Compose 및 Kubernetes를 이용한 Label Studio ML Backend를 배포하는 것을 시도했으나 `predict()`는 잘 수행하지만 `fit()`을 수행하지 못하는 현상에 의해 실패했다.
아래 code는 Docker Compose를 통해 Label Studio ML Backend를 배포하는 code인데 위의 예시들과 같이 `Start Training`을 눌러서 `fit()`을 수행하려 했지만 log 조차 출력되지 않았다.

<details>
<summary>
전체 code
</summary>

```python model.py
import os
import shutil

import numpy as np
import zerohertzLib as zz
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys, is_skipped
from PIL import Image
from ultralytics import YOLO

logger = zz.logging.Logger("LS_ML_BE")


class YOLOv8Det(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(YOLOv8Det, self).__init__(**kwargs)
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, "RectangleLabels", "Image"
        )
        logger.info("-" * 30)
        logger.info(f"Train Output: {self.train_output}")
        MODEL_FILE = self.train_output.get("model_file") or os.environ.get("MODEL_FILE")
        if MODEL_FILE:
            logger.info("Load: " + MODEL_FILE)
            self.model = YOLO(MODEL_FILE)
        else:
            logger.info("Init: YOLOv8!")
            self.model = YOLO("yolov8l.pt")
        logger.info(f"from_name: {self.from_name}")
        logger.info(f"to_name: {self.to_name}")
        logger.info(f"value: {self.value}")
        logger.info(f"classes: {self.classes}")
        logger.info("-" * 30)

    def _get_image(self, path):
        return path.replace("/data/", "./data/media/")

    def predict(self, tasks, **kwargs):
        task = tasks[0]
        image = Image.open(self._get_image(task["data"][self.value]))
        predictions = []
        score = 0
        original_width, original_height = image.size
        results = self.model.predict(image)
        i = 0
        for result in results:
            for i, prediction in enumerate(result.boxes):
                xyxy = prediction.xyxy[0].tolist()
                predictions.append(
                    {
                        "id": str(i),
                        "from_name": self.from_name,
                        "to_name": self.to_name,
                        "type": "rectanglelabels",
                        "score": prediction.conf.item(),
                        "original_width": original_width,
                        "original_height": original_height,
                        "image_rotation": 0,
                        "value": {
                            "rotation": 0,
                            "x": xyxy[0] / original_width * 100,
                            "y": xyxy[1] / original_height * 100,
                            "width": (xyxy[2] - xyxy[0]) / original_width * 100,
                            "height": (xyxy[3] - xyxy[1]) / original_height * 100,
                            "rectanglelabels": [
                                self.classes[
                                    int(prediction.cls.item()) % len(self.classes)
                                ]
                            ],
                        },
                    }
                )
                score += prediction.conf.item()
        return [
            {
                "result": predictions,
                "score": score / (i + 1),
                "model_version": os.environ.get("MODEL_FILE", "Vanilla"),
            }
        ]

    def _make_dataset(self, completion):
        if completion["annotations"][0].get("skipped"):
            return
        if completion["annotations"][0].get("was_cancelled"):
            return
        if is_skipped(completion):
            return
        file_name = completion["data"][self.value].split("/")[-1]
        shutil.copy(
            self._get_image(completion["data"][self.value]),
            f"./data/train/images/{file_name}",
        )
        file_name = ".".join(file_name.split(".")[:-1]) + ".txt"
        annotations = []
        for result in completion["annotations"][0]["result"]:
            cls = self.classes.index(result["value"]["rectanglelabels"][0])
            pts = (
                np.array(
                    [
                        result["value"]["x"] + result["value"]["width"] / 2,
                        result["value"]["y"] + result["value"]["height"] / 2,
                        result["value"]["width"],
                        result["value"]["height"],
                    ]
                )
                / 100
            )
            pts = " ".join(map(str, pts.tolist()))
            annotations.append(f"{cls} {pts}")
        with open(f"./data/train/labels/{file_name}", "w") as file:
            file.writelines("\n".join(annotations))

    def fit(self, completions, event, **kwargs):
        logger.info(f"Event: {event}")
        if event in self.TRAIN_EVENTS:
            return {"model_file": os.environ.get("MODEL_FILE")}
        zz.util.rmtree("./data/train/labels")
        zz.util.rmtree("./data/train/images")
        logger.info("Train: Start!")
        for completion in completions:
            self._make_dataset(completion)
        results = self.model.train(data="data.yaml", epochs=200, imgsz=640, device="0")
        logger.info("Train: Done!\t" + f"[{results.save_dir}/weights/best.pt]")
        os.environ["MODEL_FILE"] = f"{results.save_dir}/weights/best.pt"
        return {
            "model_file": f"{results.save_dir}/weights/best.pt",
        }
```

```docker Dockerfile
FROM python:3.8-slim

ENV PYTHONUNBUFFERED=True \
    PORT=9090
ENV YOLO_CONFIG_DIR=/app/Ultralytics

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

CMD exec gunicorn --preload --bind :$PORT --workers 1 --threads 8 --timeout 0 --log-level debug _wsgi:app
```

```yaml docker-compose.yml
version: "3.8"

services:
  redis:
    image: redis:alpine
    container_name: redis
    hostname: redis
    volumes:
      - "./data/redis:/data"
    expose:
      - 6379
    ports:
      - 6379:6379
  server:
    build: .
    container_name: server
    environment:
      - MODEL_DIR=/data/models
      - RQ_QUEUE_NAME=default
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - LABEL_STUDIO_USE_REDIS=true
      - LABEL_STUDIO_ML_BACKEND_V2=True
    depends_on:
      - redis
    links:
      - redis
    volumes:
      - "./data/server:/app/data"
      - "./data/logs:/tmp"
  label-studio:
    image: heartexlabs/label-studio
    container_name: label-studio
    depends_on:
      - redis
    ports:
      - 8080:8080
    volumes:
      - "./data/server:/label-studio/data"
    user: "1000"
```

</details>

이를 해결하기 위해 `Settings` > `Cloud Storage` > `Add Source Storage` > `Redis`를 시도했지만 오류가 발생하여 `LABEL_STUDIO_USE_REDIS=false`로 선언하여 해결했다.

</details>

<details>
<summary>
<code>RuntimeError: DataLoader worker (pid(s) *) exited unexpectedly</code>
</summary>

공유 memory 크기 제한으로 발생하는 문제이기 때문에 아래와 같이 공유 memory를 확장하여 해결한다.

```yaml backend.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
spec:
  template:
    ...
    spec:
      ...
      containers:
        - name: backend
          ...
          volumeMounts:
            ...
            - name: dshm
              mountPath: "/dev/shm"
      volumes:
        ...
        - name: dshm
          emptyDir:
            medium: Memory
```

</details>
