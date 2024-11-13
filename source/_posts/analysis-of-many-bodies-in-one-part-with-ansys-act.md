---
title: Analysis of Many Bodies in One Part with ANSYS ACT
date: 2022-03-29 13:17:15
categories:
- Etc.
tags:
- Python
- Mechanical Engineering
---
# Introduction

> 한 Part의 각 Body에 상응하는 조건을 정의하고 해석하기 위해 진행

<img width="600" alt="Introduction" src="https://user-images.githubusercontent.com/42334717/160532674-71a42534-d381-4964-99d4-d21d6adcbb17.png">

1. Mechanical - Engineering Data: 물질에 대한 성질 정의
2. Mecahnical - Geometry: 한 Part에 많은 Body 생성 (Python, SpaceClaim)
3. Mechanical - Model: Mesh 생성
4. Mechanical - Setup: 각 Body에 상응하는 경계조건 정의 (Python)
5. Mechanical - Solution: 해석

<!-- More -->

***

# Generating Many Bodies in One Part

위 사진과 같이 수많은 Body들을 직접 모델링하는 것은 불가능하기 때문에 `SpaceClaim` 내의 기능인 `Scripting`을 사용한다.
`SpaceClaim`의 `Scripting` 기능은 `IronPython`을 기반으로 Coding 및 실행이 가능하다.

~~~Python MakeManyBodiesInOnePart.py
def MakeBody(startPos, BodySize, Thickness):
    selection = Part1
    result = ComponentHelper.CreateNewComponent(selection, Info1)
    result = SketchHelper.StartConstraintSketching()
    plane = Plane.PlaneXY
    result = ViewHelper.SetSketchPlane(plane)
    point1 = Point2D.Create(MM(startPos[0]),MM(startPos[1]))
    point2 = Point2D.Create(MM(startPos[0]+BodySize),MM(startPos[1]))
    point3 = Point2D.Create(MM(startPos[0]+BodySize),MM(startPos[1]+BodySize))
    result = SketchRectangle.Create(point1, point2, point3)
    baseSel = SelectionPoint.Create(CurvePoint7)
    targetSel = SelectionPoint.Create(DatumPoint2)
    mode = InteractionMode.Solid
    result = ViewHelper.SetViewMode(mode, Info3)
    selection = Face2
    options = ExtrudeFaceOptions()
    options.ExtrudeType = ExtrudeType.Add
    result = ExtrudeFaces.Execute(selection, MM(Thickness), options, Info4)
    
BS = 4
for i in range(0, Length of Web, BS): #MD
    for j in range(0, Width of Web, BS): #CMD
        MakeBody([i, j], BS, 0.1)
~~~

<img width="960" alt="Many Bodies in One Part" src="https://user-images.githubusercontent.com/42334717/160554712-5301718b-2ce9-4b8f-837d-9421d6f87964.png">

+ `Share Mophology - Share` 필수

<img width="500" alt="Share Mophology - Share" src="https://user-images.githubusercontent.com/42334717/160571077-35c442fd-d59c-4fc5-9364-336cb0cdda11.png">

***

# Setting Condition in Many Bodies

수많은 Body들의 초기조건 및 경계조건을 정의하기 위해 `Ansys Mechanical` 내의 기능인 `Scripting`을 사용한다.
`Ansys Mechanical`의 `Scripting` 기능은 `IronPython`을 기반으로 Coding 및 실행이 가능하다.

~~~Python SetPressureCondition.py
bodies = ExtAPI.DataModel.Tree.GetObjectsByType(DataModelObjectCategory.Body)

MD = 125
CMD = 38

Pressure = 2.7*9.81 / (0.1*150)

pressure_condition = DataModel.AnalysisList[0].AddPressure()
selection = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
s1 = [bodies[0].GetGeoBody().Faces[3]]
for i in range(1,CMD,1):
    s1.append(bodies[i].GetGeoBody().Faces[3])
selection.Entities = s1
pressure_condition.Location = selection
pressure_condition.Magnitude.Output.SetDiscreteValue(0, Quantity(-Pressure, "MPa"))

pressure_condition = DataModel.AnalysisList[0].AddPressure()
selection = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
s2 = [bodies[MD*CMD-CMD].GetGeoBody().Faces[2]]
for i in range(MD*CMD-CMD+1,MD*CMD,1):
    s2.append(bodies[i].GetGeoBody().Faces[3])
selection.Entities = s2
pressure_condition.Location = selection
pressure_condition.Magnitude.Output.SetDiscreteValue(0, Quantity(-Pressure, "MPa"))
~~~

~~~Python SetThermalCondition.py
import csv

bodies = ExtAPI.DataModel.Tree.GetObjectsByType(DataModelObjectCategory.Body)

MD = 125
CMD = 38

temp_csv = csv.reader(open('Temp.csv','r'))

temp = []

for t in temp_csv:
    tmp = float(t[2])
    temp.append(tmp)

x = 1

for j in range(0,MD,1):
    for i in range(0,CMD,1):
        thermal_condition = DataModel.AnalysisList[0].AddThermalCondition()
        selection = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
        selection.Entities = [bodies[x-1].GetGeoBody()]
        thermal_condition.Location = selection
        thermal_condition.Magnitude.Output.SetDiscreteValue(0, Quantity(temp[x-1], "C"))
        x = x + 1
~~~

<img width="960" alt="Setting Thermal Condition" src="https://user-images.githubusercontent.com/42334717/160555986-ef8502bb-ae5f-4a6f-a0af-e4a60b367f40.png">
