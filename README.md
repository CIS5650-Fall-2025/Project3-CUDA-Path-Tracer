# CUDA Path Tracer
##### Example Render:
<p align="center">
  <img src="img/HeroRender.png" width="1000" />
</p>

###### 2350 x 1000 | CUDA Path Tracer with Intel OIDN Denoiser | BVH | 500 SPP | All assets modelled in Maya

<p align="center">
  <img src="img/Breakdown.png" width="500" />
</p>

---

### University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3
* Logan Cho
  * [LinkedIn](https://www.linkedin.com/in/logan-cho/)
  * [Art / Coding Portfolio!](https://www.logancho.com/)
* Run on: Windows 11, 13th Gen Intel(R) Core(TM) i7-13700H, 2.40 GHz, RTX 4060 Laptop GPU
-----

* [Intro](#Introduction)
* [Features](#Features)
* * [BSDFs](#BSDFs)
* * [Integrators](#Integrators)
* * [Mesh and Texture Loading](#Mesh)
* * [Denoiser](#OIDN)
* [Perf Analysis](#Perf-Analysis)

-----

## Introduction

This project is a CUDA Path Tracer developed on top of the base code provided by the University of Pennsylvania, CIS 565: GPU Programming and Architecture. Features implemented include: 
* Parallelized Naive Lighting Integrator
* Parallelized Full Lighting Integrator with MIS
* Performance Optimizations
  * Sorting Path Segments by material type to reduce divergence and increase warp occupancy
  * Stream Compaction of Path Segments to free up cores from rays that have already terminated
* Arbitrary .glTF mesh support with materials and textures (albedo and normals)
* BSDFs (Diffuse, Specular, Microfacet, etc.)
* Intel OIDN Integration (Denoiser)

## Features

### BSDFs

* #### Diffuse

<p align="center">
  <img src="img/cornell/4.png" width="300" />
</p>

* #### Specular
<p align="center">
  
  | <img width="300px" src="img/cornell/naive.png"> | <img width="300px" src="img/cornell/1.png"> | <img width="300px" src="img/sonic/11.png"> |
  |:--:|:--:|:--:|
  
</p>

* #### Microfacet

<p align="center">
  
  | <img width="300px" src="img/cornell/3.png"> | <img width="300px" src="img/sonic/1.png"> |
  |:--:|:--:|
  
</p>

* #### Ceramic / Plastic

<p align="center">
  
  | <img width="300px" src="img/cornell/2.png"> | <img width="300px" src="img/sonic/9.png"> |
  |:--:|:--:|
  
</p>

---

### Integrators

* #### Naive Integrator

* #### Direct Lighting Integrator

* #### Full Lighting Integrator

---

### Mesh

---

### OIDN


## Perf Analysis

### Sort by Material

### Stream Compaction

## Bloopers

## Credits

