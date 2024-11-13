---
title: Labeling with Label Studio for Pre-labeled Data using YOLOv5
date: 2023-04-14 23:44:46
categories:
- 4. MLOps
tags:
- Docker
- Python
- Label Studio
---
# Introduction

> Label Studio

Label Studio는 오픈 소스 데이터 라벨링 툴로, 기계 학습 및 데이터 분석 작업을 위해 데이터에 레이블을 부착하는 데 사용되는 도구입니다. Label Studio를 사용하면 이미지, 텍스트, 오디오, 비디오 등 다양한 유형의 데이터에 대해 레이블을 지정하고, 데이터를 분류, 개체 감지, 개체 분류, 개체 추출, 텍스트 분석 등의 작업을 수행할 수 있습니다.
Label Studio는 다양한 데이터 형식 및 레이블링 작업에 대한 다양한 템플릿을 제공하며, 커스텀 레이블링 작업을 설정할 수 있습니다. 또한, 데이터 라벨링 작업을 위해 다양한 작업자를 관리하고, 작업 상태를 추적하며, 결과를 검토하고 승인할 수 있는 기능들을 제공합니다.
Label Studio는 웹 기반 인터페이스를 제공하여 다양한 사용자가 쉽게 접근하고 사용할 수 있습니다. 또한, API를 통해 레이블링 작업을 자동화하고, 다양한 형식의 데이터를 가져와 레이블링을 수행할 수 있습니다.
Label Studio는 머신러닝, 딥러닝, 자연어 처리, 컴퓨터 비전 등 다양한 분야에서 데이터 라벨링 작업을 수행하는 데 유용하게 사용될 수 있습니다. 오픈 소스로 제공되기 때문에, 사용자들은 커스터마이징이 가능하고 개발자들은 소스 코드를 수정하여 자신의 요구에 맞게 확장할 수 있습니다.

<!-- More -->

본 글에서는 1200장의 데이터에 대해 Detection을 위한 Bbox Annotation을 손쉽게 하기위해 Pre-labeling 방법을 사용할 것이다.

> Pre-labeling

Pre-labeling은 데이터 라벨링 작업을 수행하기 전에, 초기 레이블 또는 예비 레이블을 데이터에 부착하는 작업을 의미합니다. 이는 라벨링 작업의 효율성을 높이고, 라벨링 작업자의 작업 부담을 줄이는 데 도움이 됩니다.
데이터 라벨링은 기계 학습 및 딥러닝 모델을 훈련시키기 위해 필요한 작업으로, 데이터에 레이블 또는 태그를 부착하여 모델이 원하는 결과를 예측하도록 하는 과정입니다. 그러나 라벨링 작업은 보통 시간과 노력이 많이 소요되며, 대량의 데이터에 대해 레이블을 부착하는 작업은 번거로울 수 있습니다.
이에 따라, pre-labeling은 초기 레이블 또는 예비 레이블을 데이터에 자동으로 부착하는 방법으로, 인공지능 기술을 활용하여 데이터에 대한 예비 레이블을 생성하거나, 이미지 처리, 텍스트 처리, 오디오 처리 등의 기술을 사용하여 초기 레이블을 부착합니다. 이렇게 부착된 초기 레이블은 라벨링 작업자에게 제공되어, 라벨링 작업을 수행할 때 참고하거나 수정할 수 있습니다.
pre-labeling은 라벨링 작업의 효율성을 향상시키고, 라벨링 작업자의 작업 부담을 줄이는 장점이 있습니다. 초기 레이블 또는 예비 레이블은 라벨링 작업의 출발점으로 활용되며, 라벨링 작업자는 이를 기반으로 추가적인 레이블링을 수행하여 최종 레이블을 완성할 수 있습니다. 또한, pre-labeling은 라벨링 작업의 일관성과 품질을 향상시키는 데도 도움이 될 수 있습니다.

요약하자면 아래와 같은 순서로 진행된다.

1. 다량의 데이터 중 소수만을 Annotation
2. 최종적으로 학습할 모델을 Annotation이 완료된 데이터로 학습
3. 나머지 데이터에 대해 학습된 모델로 Inference
4. Inference 값들을 더욱 정확하게 수정하여 다량의 데이터에 대해 최종 Annotation
5. 완성된 데이터로 최종 모델 학습!

본 글에서는 모델은 [YOLOv5](https://github.com/ultralytics/yolov5)를 사용했으며 `3.`과 `4.` 사이에서 어떻게 Inference 데이터를 Label Studio에 업로드하기 위해 변환하는지, 그리고 Label Studio 상에서 어떻게 수정할 수 있게 설정하는지를 다뤄볼 예정이다.

---

# From YOLOv5 to Label Studio

학습된 YOLOv5 모델의 출력은 아래 예시와 같다.

```json YOLOv5_Inference_Example.txt
0 0.525805 0.49504 0.561043 0.990079
```

좌에서 우로 `label`, `x`, `y`, `w`, `h`를 의미한다.
하지만 소량의 데이터로 학습된 모델로 생성한 데이터이기 때문에 간혹 어떤 대상도 Detection 하지 못하는 경우가 발생한다.
그렇기 때문에 아래의 코드를 통해 보완해줄 수 있다.

```python bin.py
import os


ORG = os.getcwd()
for tmp in ["train", "val", "test"]:
    os.chdir(ORG + '/detect/' + tmp)
    IMG = os.listdir()
    TXT = os.listdir('labels')
    for I in IMG:
        if ".png" in I:
            if not I.replace('png', 'txt') in TXT:
                with open('labels/' + I.replace('png', 'txt'), 'w') as f:
                    f.write("0 0.5 0.5 1 1")
                print("ON")
    print(len(IMG))
    print(len(TXT))
```

이렇게 Pre-labeling이 완료된 `.txt` 파일을 가지고 있으면 이를 `.json`으로 규칙에 맞춰 변화시켜야 Label Studio에서 사용할 수 있다.

```python yolo2ls.py
import os
import json

from skimage import io


def yolo2ls(home):
    '''
    home: path (str)
    ├── test
    │   └── labels
    ├── train
    │   └── labels
    └── val
        └── labels
    '''
    tmp = os.getcwd()
    os.chdir(home)
    for status in ["train", "val", "test"]:
        res = []
        os.chdir(home)
        os.chdir(status)
        for i in os.listdir():
            if ".png" in i:
                #----------- JSON -----------#
                obj = {"data": {"image": f"/data/local-files/?d={status}/{i}"}}
                #----------- IMG META -----------#
                img = io.imread(i)
                original_height, original_width, c = img.shape
                #----------- Pre-labeling -----------#
                gtf = "labels/" + i.replace("png", "txt")
                with open(gtf, "r") as f:
                    gt = f.readlines()
                result_list = []
                for j in gt:
                    cx, cy, w, h = [float(k) for k in j.strip().split(" ")[1:]]
                    xmin = max(0.0, round(100.0 * (cx - w / 2), 2))
                    ymin = max(0.0, round(100.0 * (cy - h / 2)))
                    xmax = min(100.0, round(100.0 * (cx + w / 2)))
                    ymax = min(100.0, round(100.0 * (cy + h / 2)))
                    points = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
                    result_list.append(
                            {"original_width": original_width,
                            "original_height": original_height,
                            "image_rotation": 0,
                            "value": {
                                "points": points,
                                "closed": True,
                                "polygonlabels": ["receipt"]
                            },
                            "from_name": "label",
                            "to_name": "image",
                            "type": "polygonlabels",
                            "origin": "manual"}
                        )
                pred = [{
                    "result": result_list
                }]
                obj.update({"annotations": pred})
                res.append(obj)
        os.chdir(tmp)
        with open(f"./pre-labeling-{status}.json", "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    home = ${home}
    yolo2ls(home)
```

---

# Upload to Label Studio

```docker Dockerfile
FROM heartexlabs/label-studio

ENV LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
ENV LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/user
```

```bash build.sh
docker build -t ${image_name} .
docker run \
    --name ${container_name} \
    -p 8080:8080 \
    -v ${DS_ROOT}/data:/label-studio/data \
    -v ${DS_ROOT}/files:/home/user \
    ${image_name}
```

![Local files 1](https://user-images.githubusercontent.com/42334717/231914244-163c8e3b-c6f3-48e9-85da-a7f87fbb2002.png)

![Local files 2](https://user-images.githubusercontent.com/42334717/231919586-7aed66ca-e643-4d65-972a-c7656e2f7cfc.png)

이렇게 설정을 마치고 `yolo2ls.py`에서 생성된 `.json` 파일을 업로드해주면 아래와 같이 Pre-labeling이 완료된 Repository를 확인할 수 있다.

![Pre-labeling 1](https://user-images.githubusercontent.com/42334717/231921155-0dc7c02c-7fdc-4d5e-bfd0-450f453b5157.png)

![Pre-labeling 2](https://user-images.githubusercontent.com/42334717/231982165-6551757b-11bf-4cc7-9bc7-2a384cbb95d5.png)

---

# Etc.

<details>
<summary>
이미지가 메타 데이터에 의해 회전된 경우
</summary>

```python rot2org.py
import os
import shutil
from PIL import Image

from tqdm import tqdm


def isRot(home, imgpath):
    img = Image.open(imgpath)
    meta = img._getexif()
    try:
        if meta[274] == 6:
            img = img.rotate(-90)
        img.save(home.replace("rot", "DS") + '/' + imgpath)
    except:
        shutil.copy(imgpath, home.replace("rot", "DS") + '/' + imgpath)

if __name__ == "__main__":
    h = ['datasets/rot/train',
    'datasets/rot/val',
    'datasets/rot/test']
    for home in h:
        os.chdir(home)
        for i in tqdm(os.listdir()):
            if "png" in i:
                isRot(home, i)
```

</details>

[Set up connection in the Label Studio UI](https://labelstud.io/guide/storage.html#Set-up-connection-in-the-Label-Studio-UI-4)