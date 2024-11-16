---
title: '전문연구요원: 업체 정보'
date: 2023-09-04 14:19:23
categories:
- 0. Daily
tags:
- 전문연구요원
- Python
- Selenium
- CI/CD
- GitHub
- GitHub Actions
---
# Introduction

박사 전문연구요원은 TEPS와 한국사능력검정시험을 응시하고 점수에 따라 자신의 학교에서 복무하면 된다.
하지만 석사 전문연구요원은 편입 과정이 박사 전문연구요원과는 다르게 별도의 시험을 응시하는 것이 아닌, 일반 취업과 같다.
물론 TO를 가지고 있는 기업만 편입이 가능하다.
아마 석사를 갓 졸업한 대학원생이고, 주위에 전문연구요원으로 재직하고 있는 선배 혹은 지인이 없다면 매우 막막할 것이라 생각한다.
편입을 위해 업체를 찾아볼 때 전문연구요원을 채용할 수 있는지, 자신의 분야와 맞는지, 복지는 어떤지, 연봉은 어떤지 등 고려할 것이 너무 많아 더욱 어려울 것이다.
회사 공식 페이지에는 구인 글이 존재하지만 사람인이나 wanted와 같은 플랫폼에 구인글이 올라오지 않아 접근성이 떨어지는 경우 TO를 보유한 기업 혹은 전문연구요원 복무자가 존재하는 업체 리스트를 [산업지원 병역일터](https://work.mma.go.kr/caisBYIS/search/byjjecgeomsaek.do?menu_id=m_m6_1)에서 확인할 수 있지만, 매번 다운로드하고 엑셀에서 일일히 찾아볼 수 없는 일이다.
그래서 아래와 같이 전문연구요원 TO를 가지고 있는 업체의 통계적 시각화와 시간에 따른 복무 인원 변화를 제공하는 프로젝트를 진행하게 됐다.
GitHub Actions로 자동화되어 10일에 한번 데이터 및 시각화 자료가 갱신된다.

<div align = "center">
  <a href="https://github.com/Zerohertz/awesome-jmy">
    <img src="https://img.shields.io/badge/awesome--jmy-800a0a?style=for-the-badge&logo=Awesome Lists&logoColor=white"/>
  </a>
</div>

이 프로젝트에서 얻거나 확인할 수 있는 것들은 아래와 같다.

1. 시간에 따른 전문연구요원 업체의 데이터 (`data/병역지정업체검색_YYYYMMDD.xls`)
2. 현재 전문연구요원 업체의 분야, 위치 등 시각화
3. 시간과 업체에 따른 복무 인원 데이터 (`prop/time.tsv`) 및 시각화

<!-- More -->

---

# Python

## Data Download

아래 코드를 통해 현재 시점의 데이터를 `selenium`로 다운로드할 수 있다.

```python src/crawling.py
import json
import os
import time

import requests
from selenium import webdriver


class download_data:
    def __init__(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
        )
        self.browser = webdriver.Chrome(options)
        self.webhook = os.environ.get("WEBHOOK")

    def main(self):
        # 산업지원 병역일터 접속
        self.browser.get(
            "https://work.mma.go.kr/caisBYIS/search/byjjecgeomsaek.do?eopjong_gbcd_yn=1&eopjong_gbcd=2"
        )
        # Data Download
        self._xpath_click(
            "/html/body/div[1]/div[2]/div[2]/div/div/div[2]/div[2]/div[1]/span/a",
        )
        time.sleep(30)

    def _xpath_click(self, element):
        element = self.browser.find_element("xpath", element)
        element.click()

    def _send_discord_message(self, content):
        data = {"content": content}
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.webhook, data=json.dumps(data), headers=headers)
        return response
```

## Visualization

다운로드한 `병역지정업체검색_YYYYMMDD.xls` 데이터를 읽고 시각화할 수 있게 아래 코드를 개발했다.

```python src/vis.py
import os
import warnings
from glob import glob

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")


class vis_data:
    def __init__(self, file_name, data, degree):
        self.time = file_name[-12:-4]
        self.data = data
        """
        NOTE: "벤처기업부설연구소", "중견기업부설연구소", "중소기업부설연구소"를 제외한 모든 업종은 박사 전문연구요원으로 간주
        과기원
        과기원부설연구소
        국가기관 등 연구소
        기초연구연구기관
        대기업부설연구소
        대학원연구기관
        방산연구기관
        벤처기업부설연구소
        자연계대학부설연구기관
        정부출연연구소
        중견기업부설연구소
        중소기업부설연구소
        지역혁신센터연구소
        특정연구소
        """
        os.makedirs("prop", exist_ok=True)
        DIR_NAME = ["ALL", "MS", "PhD"]
        self.degree = DIR_NAME[degree]
        self.dir = os.path.join("prop", DIR_NAME[degree])
        os.makedirs(self.dir, exist_ok=True)
        if degree == 1:
            self.data = self.data[
                (self.data["업종"] == "벤처기업부설연구소")
                | (self.data["업종"] == "중견기업부설연구소")
                | (self.data["업종"] == "중소기업부설연구소")
            ]
        elif degree == 2:
            self.data = self.data[
                ~(
                    (self.data["업종"] == "벤처기업부설연구소")
                    | (self.data["업종"] == "중견기업부설연구소")
                    | (self.data["업종"] == "중소기업부설연구소")
                )
            ]
        self.data["위치"] = (
            self.data["주소"]
            .str.replace("서울특별시 ", "서울특별시")
            .str.replace("경기도 ", "경기도")
            .str.split(" ")
            .str[0]
            .str.replace("서울특별시", "서울특별시 ")
            .str.replace("경기도", "경기도 ")
        )
        self.ranked_data_org = self.data.sort_values(
            by=["현역 복무인원", "현역 편입인원", "업체명"], ascending=[False, False, True]
        ).iloc[:, [1, 14, 15, 16]]
        self.ranked_data_new = self.data.sort_values(
            by=["현역 편입인원", "현역 복무인원", "업체명"], ascending=[False, False, True]
        ).iloc[:, [1, 14, 15, 16]]
        plt.rcParams["font.size"] = 15
        plt.rcParams["font.family"] = "Do Hyeon"

    def time_tsv(self):
        print("WRITE TIME SERIES TSV")
        with open(f"prop/time.tsv", "a") as f:
            for name, _, a, b in self.ranked_data_org.values:
                f.writelines(f"{self.time}\t{name}\t{a}\t{b}\n")

    def pie_hist(self, tar, threshold=3):
        print("PLOT PIE & HIST:\t", tar)
        field_counts = self.data[tar].value_counts()
        large_parts = field_counts[field_counts / len(self.data) * 100 >= threshold]
        small_parts = field_counts[field_counts / len(self.data) * 100 < threshold]
        large_parts_labels = [
            f"{i} ({v})" for i, v in zip(large_parts.index, large_parts.values)
        ]
        plt.figure(figsize=(30, 10))
        plt.subplot(1, 2, 1)
        colors = sns.color_palette("coolwarm", n_colors=len(large_parts))[::-1]
        plt.pie(
            large_parts,
            labels=large_parts_labels,
            autopct="%1.1f%%",
            startangle=90,
            radius=1,
            colors=colors,
        )
        plt.title(f"{threshold}% 이상 {tar} 분포", fontsize=25)
        plt.subplot(1, 2, 2)
        plt.grid(zorder=0)
        small_parts = small_parts[:15]
        colors = sns.color_palette("Spectral", n_colors=len(small_parts))
        bars = plt.bar(
            small_parts.index,
            small_parts.values,
            color=colors[: len(small_parts)],
            zorder=2,
        )
        for bar in bars:
            height = bar.get_height()
            percentage = (height / len(self.data)) * 100
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{percentage:.1f}%",
                ha="center",
                va="bottom",
            )
        plt.xlabel(tar)
        plt.ylabel("빈도")
        plt.xticks(small_parts.index, rotation=45)
        plt.title(f"{threshold}% 미만 {tar} 분포", fontsize=25)
        plt.savefig(f"{self.dir}/{tar}.png", dpi=300, bbox_inches="tight")

    def rank_vis(self, by="현역 복무인원", top=30):
        print("PLOT RANK:\t", by)
        plt.figure(figsize=(10, int(0.6 * top)))
        plt.grid(zorder=0)
        colors = sns.color_palette("coolwarm", n_colors=top)
        if by == "현역 복무인원":
            bars = plt.barh(
                self.ranked_data_org["업체명"][:top][::-1],
                self.ranked_data_org[by][:top][::-1],
                color=colors,
                zorder=2,
            )
        elif by == "현역 편입인원":
            bars = plt.barh(
                self.ranked_data_new["업체명"][:top][::-1],
                self.ranked_data_new[by][:top][::-1],
                color=colors,
                zorder=2,
            )
        MAX = bars[-1].get_width()
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + MAX * 0.01,
                bar.get_y() + bar.get_height() / 4,
                f"{width}명",
                ha="left",
                va="bottom",
            )
        plt.xlabel(by)
        plt.ylabel("업체명")
        plt.xlim([0, MAX * 1.1])
        plt.title(f"{by} TOP {top}", fontsize=25)
        plt.savefig(
            f"{self.dir}/TOP_{top}_{by.replace(' ', '_')}.png",
            dpi=300,
            bbox_inches="tight",
        )

    def rank_readme(self, top=0):
        print("WRITE README.md")
        with open(f"{self.dir}/README.md", "w") as f:
            if top == 0:
                f.writelines(
                    f"<div align=center> <h1> :technologist: 전문연구요원 현역 복무인원 순위 :technologist: </h1> </div>\n\n<div align=center>\n\n|업체명|현역 배정인원|현역 편입인원|현역 복무인원|\n|:-:|:-:|:-:|:-:|\n"
                )
                for name, a, b, c in self.ranked_data_org.values:
                    f.writelines(
                        f"|[{name}](https://github.com/Zerohertz/awesome-jmy/blob/main/prop/time/{name.replace('(', '').replace(')', '').replace('/', '').replace(' ', '')}.png)|{a}|{b}|{c}|\n"
                    )
            else:
                f.writelines(
                    f"<div align=center> <h1> :technologist: 전문연구요원 현역 복무인원 순위 TOP {top} :technologist: </h1> </div>\n\n<div align=center>\n\n|업체명|현역 배정인원|현역 편입인원|현역 복무인원|\n|:-:|:-:|:-:|:-:|\n"
                )
                for name, a, b, c in self.ranked_data_org.values[:top]:
                    f.writelines(
                        f"|[{name}](https://github.com/Zerohertz/awesome-jmy/blob/main/prop/time/{name.replace('(', '').replace(')', '').replace('/', '').replace(' ', '')}.png)|{a}|{b}|{c}|\n"
                    )
            f.writelines("\n</div>")

    def plot_time(self):
        os.makedirs(f"prop/time", exist_ok=True)
        time_data = pd.read_csv(
            f"prop/time.tsv", sep="\t", header=None, encoding="utf-8"
        )
        for name in time_data.iloc[:, 1].unique():
            print("PLOT TIME SERIES:\t", name)
            self._twin_plot(time_data, name)
            plt.savefig(
                f"prop/time/{name.replace('(', '').replace(')', '').replace('/', '').replace(' ', '')}.png",
                dpi=100,
                bbox_inches="tight",
            )
            plt.close("all")

    def _twin_plot(self, data, name):
        tmp = data[data.iloc[:, 1] == name]
        x, y1, y2 = (
            pd.to_datetime(tmp.iloc[:, 0], format="%Y%m%d"),
            tmp.iloc[:, 3],
            tmp.iloc[:, 2],
        )
        _, ax1 = plt.subplots(figsize=(20, 10))
        plt.grid()
        ax1.plot(x, y1, "b--", linewidth=2, marker="o", markersize=12)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("현역 복무인원 [명]", color="b")
        ax1.tick_params("y", colors="b")
        ax2 = ax1.twinx()
        ax2.plot(x, y2, "r-.", linewidth=2, marker="v", markersize=12)
        ax2.set_ylabel("현역 편입인원 [명]", color="r")
        ax2.tick_params("y", colors="r")
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        try:
            m = self.data[self.data["업체명"] == name]["현역 배정인원"].iloc[0]
            plt.title(f"{name} (현역 배정인원: {m}명)")
        except:
            plt.title(f"{name} (현역 배정인원: X)")
```

연구분야, 업종, 위치 등과 같이 업체의 수로 나눌 수 있는 데이터는 pie chart와 histogram으로, 현역 복무인원 및 현역 편입인원에 대한 데이터는 histogram과 표로 시각화했다.

![vis](https://github.com/Zerohertz/Zerohertz/assets/42334717/571b4071-f381-479a-8be1-4a8641bd50d8)

그리고 시간에 따른 인원을 `.tsv` ($\because$ 업체명에 `,` 존재 가능)로 저장했으며 시각화 시 복무인원과 편입인원의 scale이 상이하기 때문에 아래와 같이 y축을 두 개 사용했다.

![time](https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/265370465-27e77b78-3720-4710-95e3-d29c05710ab7.png)

---

# GitHub Actions

이 모든 작업을 자동화하기 위해 GitHub Actions의 workflow를 아래와 같이 정의했다.

```yaml .github/workflows/main.yaml
name: Download List & Vis

on:
  schedule:
    - cron: "0 0 1,11,21 * *"
  workflow_dispatch:

env:
  WEBHOOK: ${{ secrets.WEBHOOK }}

jobs:
  workflow:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
        with:
          token: ${{ secrets.GH_TOKEN }}

      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: "3.x"

      - name: Install Dependency
        run: |
          sudo apt purge google-chrome-stable
          sudo apt purge chromium-browser
          sudo apt install -y chromium-browser
          pip install requests pandas xlrd selenium seaborn matplotlib
          sudo cp prop/DoHyeon-Regular.ttf /usr/share/fonts/
          sudo fc-cache -f -v
          rm ~/.cache/matplotlib -fr

      - name: Run Script
        run: |
          rm -f prop/*/*.png
          rm -f prop/*/*.md
          rm -rf prop/time
          python main.py

      - name: Move XLS File to data Directory
        run: |
          mkdir -p data/
          mv *.xls data/

      - name: Commit and Push
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add -A
          git commit -m ":bento: Update: GitHub Actions" || echo "No changes to commit"
          git push
```

---

# 맺음말

[레포지토리](https://github.com/Zerohertz/awesome-jmy)의 코드가 최신이니 위에 사용된 코드들은 참고만 해주세요!
해당 프로젝트에 건의하고 싶거나 추가하면 좋을 아이디어가 존재한다면 댓글로 남겨주시거나 [issue](https://github.com/Zerohertz/awesome-jmy/issues) 남겨주세요!
그리고 만약 도움되셨다면 star 부탁드립니다,,, ^^,,,
마지막으로 전문연구요원 모두 화이팅입니다!