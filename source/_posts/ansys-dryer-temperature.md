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

![dryer](/images/ansys-dryer-temperature/dryer.jpg)

1. Generate geometry

<img src="/images/ansys-dryer-temperature/generate-geometry.png" alt="generate-geometry" width="818" />

2. Substract dryer and web by creating boolean

<img src="/images/ansys-dryer-temperature/substract-and-rename-each-body.png" alt="substract-and-rename-each-body" width="888" />


<!-- More -->

***

# Mesh

1. Define named selection
2. Generate mesh

<img src="/images/ansys-dryer-temperature/named-selection-and-generate-mesh.png" alt="named-selection-and-generate-mesh" width="960" />

***

# Setup & Solution

1. Set model

<img src="/images/ansys-dryer-temperature/set-model.png" alt="set-model" width="238" />

2. Set contact region

<img src="/images/ansys-dryer-temperature/set-contact-region.png" alt="set-contact-region" width="600" />

3. Set boundary condition

+ B.C. of web

<img src="/images/ansys-dryer-temperature/bc-of-web.png" alt="bc-of-web" width="672" />

+ B.C. of dryer inlet

<img src="/images/ansys-dryer-temperature/bc-of-dryer-inlet.png" alt="bc-of-dryer-inlet" width="481" />

+ B.C. of web inlet

<img src="/images/ansys-dryer-temperature/bc-of-web-inlet.png" alt="bc-of-web-inlet" width="480" />

+ B.C. of coupled wall

<img src="/images/ansys-dryer-temperature/bc-of-coupled-wall-1.png" alt="bc-of-coupled-wall-1" width="617" />

<img src="/images/ansys-dryer-temperature/bc-of-coupled-wall-2.png" alt="bc-of-coupled-wall-2" width="598" />

4. Initialization & Calculation

<img src="/images/ansys-dryer-temperature/initialization.png" alt="initialization" width="1135" />