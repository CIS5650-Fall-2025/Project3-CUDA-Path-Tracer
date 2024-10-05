CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Zixiao Wang
  * [LinkedIn](https://www.linkedin.com/in/zixiao-wang-826a5a255/)
* Tested and rendered on: Windows 11, i7-12800H @ 2.40 GHz 32GB, GTX 3070TI (Laptop)

### Showcase Image

![](img/cornell.2024-10-05_21-31-59z.5000samp.png)

### Introduction

In this project, I implement a CUDA-based path tracer capable of rendering globally illuminated images quickly. The core renderer includes a shading kernel with BSDF evaluation for Ideal diffuse, perfectly specular-reflective, combination of refraction and reflection, and imperfect specular surface incorporated with roughness parameters. Further, it also explores the features of OBJ and Texture loading, Depth of Field, Environment Mapping/Lighting, BVH structure, and Denoising. The Sections below each specify the output and implementation of each feature, including performance analysis of optimization and further development direction.


### BRDF Shading (Bidirectional Reflectance Distribution Function shading)

#### Ideal diffuse shading: The image shows basic Lambertian shading in the Cornell box scene

 ![](img/cornell.2024-10-04_22-37-41z.300samp.png)

#### Ideal Specular Shading： The image shows perfect specular surface shading in the Cornell box scene

![](img//cornell.2024-10-04_22-38-54z.300samp.png)

#### Imperfect reflection: below are 4 images incorporated with different roughness. The edge of the reflection image becomes more and more blur while roughness increases
* Top left is roughness 0.0 (the light is dim because the light parameter is about 0.3 less, but it does not affect the roughness effect)
* Top right is roughness 0.3
* Low left is roughness 0.6
* Low right is roughness 0.9

<img src="img//cornell.2024-10-04_22-38-54z.300samp.png" width="400"/> <img src="img//cornell.2024-10-04_22-42-17z.300samp.png" width="400"/> <img src="img//cornell.2024-10-04_22-42-45z.300samp.png" width="400"/> <img src="img//cornell.2024-10-04_22-43-09z.300samp.png" width="400"/>

#### Refraction and Reflection: Refraction with Fresnel effects using [Schlick's approximation](https://en.wikipedia.org/wiki/Schlick's_approximation). Implementation Reference [PBRTv4 9.3](https://pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission)
* In the real world, reflection and refraction might happen at the same time in one spot. But here, we sample one pixel with 600 samples. Then, for each sample, Monte Carlo was utilized to decide on either refraction or reflection. Once averaged out, it is real.

![](img//cornell.2024-10-04_22-40-30z.600samp.png)

### OBJ Mesh Loading and Texture Mapping

<img src="img//cornell.2024-10-04_22-49-50z.350samp.png" width="300"/> <img src="img//cornell.2024-10-04_22-51-16z.350samp.png" width="300"/> <img src="img//cornell.2024-10-04_22-52-44z.350samp.png" width="300"/> 

#### Environment Map Lighting

![](img//cornell.2024-10-05_21-41-57z.500samp.png)

#### BVH accelerated spatial structure

<img src="img//5cf37bd66d3d91e8c63c6c4c3e3adc0.png" width="500"/> <img src="img//c8b3c1f59160aef0ed023c584f95c8c.png" width="500"/>

#### DOP (Depth of Field)

<img src="img//0297d40cc99ebea47bc53b43469d7ba.jpg" width="355"/> <img src="img//1cd0328771c43f1d9a24924497270b7.jpg" width="400"/>

#### Denoise with intel Open Image Denoise

<img src="img//cornell.2024-10-04_23-05-31z.350samp.png" width="400"/> <img src="img//cornell.2024-10-04_22-49-50z.350samp.png" width="400"/>


