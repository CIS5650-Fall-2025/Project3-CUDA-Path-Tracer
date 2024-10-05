CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

### Yuhan Liu

[LinkedIn](https://www.linkedin.com/in/yuhan-liu-), [Personal Website](https://liuyuhan.me/), [Twitter](https://x.com/yuhanl_?lang=en)

**Tested on: Windows 11 Pro, Ultra 7 155H @ 1.40 GHz 32GB, RTX 4060 8192MB (Personal Laptop)**

 <img src="img/cover.png" width="400"/>
This takes hours to path trace but I need to update it oops...

## Summary 

CUDA-based path tracer capable of rendering globally-illuminated images very quickly.

### Table of Contents

Core Features
* BSDF Evaluation
* Path Continuation/Termination
* Material Sort
* Stochastic-Sampled Antialiasing

Additional Elements
* Refraction
* Arbitrary Mesh Loading
* Texture Loading & Mapping
* Procedural Textures
* Bounding Volume Hierarchy
* Intel Open Image Denoiser

Bloopers Maybe

## Core Path Tracer Features

### BSDF Evaluation 

### Path Continuation/Termination 

### Material Sort

### Stochastic-Sampled Antialiasing 

 <img src="img/aa.png" width="400"/>  <img src="img/noaa.png" width="400"/>
 
 <img src="img/aa_close.png" width="400"/>  <img src="img/noaa_close.png" width="400"/>

## Additional Enhancements

### Refraction 

 <img src="img/refraction.png" width="400"/>

### Arbitrary Mesh Loading (OBJs)

 <img src="img/teapot.png" width="400"/>

### Texture Loading & Mapping (combined with OBJs)

 <img src="img/denoise.png" width="400"/>

### Procedural Textures on the GPU

 <img src="img/textures.png" width="400"/>

### BVH

### Intel Open Image Denoiser 

 <img src="img/noisy.png" width="400"/> <img src="img/denoised.png" width="400"/>

## Bloopers
