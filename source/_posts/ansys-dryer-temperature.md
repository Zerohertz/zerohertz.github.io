---
title: Simulation of Temperature Distribution of Material in Dryer
date: 2022-04-12 17:00:27
categories:
- Etc.
tags:
- C, C++
- Mechanical Engineering
---
# Geometry

![Dryer](/images/ansys-dryer-temperature/162924680-3bcc6dca-db38-4750-8dba-f77c004be0e4.jpg)

1. Generate geometry

<img width="818" alt="Generate Geometry" src="/images/ansys-dryer-temperature/162912378-970987b9-0ddb-4bf1-95b3-80659930cd9a.png">

2. Substract dryer and web by creating boolean

<img width="888" alt="Substract and Rename Each Body" src="/images/ansys-dryer-temperature/162912677-a67661e9-3367-4672-a167-9ae998c9cf15.png">


<!-- More -->

***

# Mesh

1. Define named selection
2. Generate mesh

<img width="960" alt="Named Selection and Generate Mesh" src="/images/ansys-dryer-temperature/162914676-cbeb4b52-2bbf-46a5-897c-44a7b6649ed4.png">

***

# Setup & Solution

1. Set model

<img width="238" alt="Set Model" src="/images/ansys-dryer-temperature/162919376-089ca904-06f3-4685-bf48-0771396104ab.png">

2. Set contact region

<img width="600" alt="Set Model" src="/images/ansys-dryer-temperature/162919936-ce1e5ac9-9ee4-46e0-bf44-50a5bcd9cb27.png">

3. Set boundary condition

+ B.C. of web

<img width="672" alt="BC of Web" src="/images/ansys-dryer-temperature/162920561-b661ae7f-27d5-4e79-acef-a861f79c15a2.png">

+ B.C. of dryer inlet

<img width="481" alt="BC of Dryer Inlet" src="/images/ansys-dryer-temperature/162920912-7d4ecfc9-c93b-4755-a7d8-4946b91b570e.png">

+ B.C. of web inlet

<img width="480" alt="BC of Web Inlet" src="/images/ansys-dryer-temperature/162921228-99de9c09-38d5-4b1c-a9f8-3130d2ca1431.png">

+ B.C. of coupled wall

<img width="617" alt="BC of Coupled Wall" src="/images/ansys-dryer-temperature/162920643-69723449-7a8a-45a4-adf6-c8b8c43666b5.png">

<img width="598" alt="BC of Coupled Wall" src="/images/ansys-dryer-temperature/162927696-d46e2618-c48d-4626-8f9f-2e3a4c0f1756.png">

4. Initialization & Calculation

<img width="1135" alt="Initialization" src="/images/ansys-dryer-temperature/162922868-cc41c691-d7f7-469f-8b18-1689466ac198.png">