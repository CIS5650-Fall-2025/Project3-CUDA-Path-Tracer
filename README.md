CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Christine Kneer
  * https://www.linkedin.com/in/christine-kneer/
  * https://www.christinekneer.com/
* Tested on: Windows 11, i7-13700HX @ 2.1GHz 32GB, RTX 4060 8GB (Personal Laptop)

## Part 1: Introduction

|![cornell 2024-10-06_03-36-00z 5000samp](https://github.com/user-attachments/assets/c98a8a83-12b8-498e-af44-8ca9f28be19e)|
|:--:|
|*D.Va & Meka*|

|![cornell 2024-10-06_22-05-35z 5000samp](https://github.com/user-attachments/assets/5d19131a-a646-4a3d-b858-afb0eeb62b38)|
|:--:|
|*Meka*|

|![cornell 2024-10-06_22-55-51z 5000samp](https://github.com/user-attachments/assets/799935af-e51a-4f2e-96ba-cfa50d2cbdd1)|![cornell 2024-10-06_22-58-11z 5000samp](https://github.com/user-attachments/assets/2fc4a70e-ff67-4d77-87c8-ea381bbb0657)|
|:--:|:--:|

|![cornell 2024-10-06_22-55-51z 5000samp](https://github.com/user-attachments/assets/f6d19f87-7eed-4d7a-9a15-4c08e20369b7)|![cornell 2024-10-06_22-58-11z 5000samp](https://github.com/user-attachments/assets/9c97f0a1-03e8-43af-b3b4-e91aa7ac7d5d)|
|:--:|:--:|


In this project, I implemented a CUDA-based path tracer capable of rendering globally-illuminated images. The path tracer incorporates the following features:
* Core
  * BSDF evaluation: Perfectly Specular, Perfectly Diffuse
  * Path continuation/termination using Stream Compaction
  * Material sorting on intersections
  * Stochastic sampled antialiasing
* Extra
  *  BSDF evaluation: Refraction, Microfacet
  *  Hierarchical spatial data structures: BVH
  *  Russian roulette path termination
  *  Arbitrary mesh loading, with texture mapping and mtl file loading (OBJ)
  *  Environment map
  *  Physically-based depth-of-field
  *  Open Image AI Denoiser
  
## Part 2: Features

In this part, we will go over the most important features in this path tracer.

### Part 2.1: BSDF evaluation

**BSDF** stands for **Bidirectional Scattering Distribution Function**. It generalizes how light is scattered by a material and can include both reflection and transmission of light. This path tracer supports the following material BSDFs: diffuse, specular, refractive (like glass), and microfacets.

|![cornell 2024-10-06_22-58-11z 5000samp](https://github.com/user-attachments/assets/e7de5683-56dc-4575-8944-a5ff0f748c33)|![cornell 2024-10-06_22-55-51z 5000samp](https://github.com/user-attachments/assets/4b399a98-8618-4ead-a29f-489be68caa3e)|![cornell 2024-10-06_23-04-01z 5000samp](https://github.com/user-attachments/assets/88f4c76f-154d-4df8-8fd7-36e443fa9f5c)|
|:--:|:--:|:--:|
|*Diffuse*|*Specular*|*Refractive*|

**Microfacets** are a concept used to model how light interacts with rough surfaces at a microscopic level. Rather than treating surfaces as perfectly smooth, microfacet models assume that a surface is made up of many tiny facets, each reflects light in different directions. In mircofacet model, **roughness** is used to descirbe how rough the surface is.

|![cornell 2024-10-06_22-05-35z 5000samp](https://github.com/user-attachments/assets/8fd7292e-8e43-4780-985d-7db4847b6985)|
|:--:|
|*Microfacets with roughness from 0.1 - 0.9*|

### Part 2.2: Stochastic sampled antialiasing

**Stochastic sampled antialiasing** is a technique used to reduce aliasing (jagged edges) in rendered images by oversampling pixels using randomly jittered sample points. Instead of sampling the color at the center of each pixel, stochastic SSAA takes several samples per pixel, each with a slight random offset (jitter), producing smoother edges and reducing artifacts.

*render here*

### Part 2.3: Hierarchical spatial data structures

**Bounding Volume Hierarchy** (BVH) is a spatial data structure used in ray tracing to efficiently speed up the process of finding ray-object intersections. It groups objects into a hierarchy of nested bounding volumes, allowing for quick rejection of large sets of objects that don't intersect with a ray, thus reducing the number of intersection tests from O(n) (naively iterate through all objects in the scene) to O(log(n)) (efficiently iterate through the tree of bounding volumes).

|![image](https://github.com/user-attachments/assets/fe4fec66-db4b-4b0e-83bb-5332bdb6a2e4)|
|:--:|
|*Example of bounding volume hierarchy, Schreiberx - Own work, from: https://en.wikipedia.org/wiki/Bounding_volume_hierarchy#/media/File:Example_of_bounding_volume_hierarchy.svg*|

Later in Part 3, we will compare the performance of our path tracer with and without BVH.

### Part 2.4: Arbitrary mesh loading

This path tracer supports OBJ mesh and optionally MTL file loading. OBJ is widely supported and used in 3D modeling software and game engines to describe the shape of 3D objects, whereas the MTL file format is used to store material properties for objects in an associated OBJ file.

OBJ meshes are loaded into the path tracer as triangles, which are then used to construct the BVH tree. At each intersection, uvs and normals are interpolated using Barycentric Interpolation, which allows us to sample the texture and shade the mesh properly.

<p align="center">
<img width="600" alt="cornell 2024-10-06_23-09-46z 3004samp" src="https://github.com/user-attachments/assets/0d828b30-0f88-494b-a452-95ac04264e34">
</p>

MTL file specifices the diffuse texture or albedo of the mesh, with other properties like **Ks** (Specular color) & **Ns** (Shininess), which gets translated into BSDF propeties supported by our path tracer. For example, shininess is mapped to roughness in our microfacet model, where higher shininess means that the material is less rough.

<p align="center">
<img width="600" alt="Screenshot 2024-10-06 164438" src="https://github.com/user-attachments/assets/249f500b-8618-43b7-860c-1640c95e799b">
</p>



### Part 2.5: Environment map

When a ray escapes the scene without hitting an object, it is considered to have hit the environment map. The direction of the ray is used to sample the environment texture to determine the color of the background or reflections. This path tracer uses spherical mapping, where the environment is treated as if it's projected onto a sphere around the scene.

*render here*

### Part 2.6: Physically-based depth-of-field

Physically-based Depth of Field simulates the way real cameras blur objects that are not at the point of focus. It is implemented by jittering rays within an aperture, which creates the blurry effect.

*render here*

### Part 2.7: Open Image AI Denoiser

This path tracer uses [OIDN](https://github.com/RenderKit/oidn), which is an open-source image denoiser that works by applying a filter on Monte-Carlo-based pathtracer output. Using a denoiser allows the render to appear at a higher quality with fewer iterations.

*render here*

## Part 3: Performance Analysis

In this part, we discuss the performance of our path tracer under different performance improvement techiniques.

### Part 3.1: Stream compaction

Stream compaction is an optimization technique that works by reducing unnecessary computations related to rays that are no longer active. At each iteration, we "remove" inactive rays from the list of active rays, effectively spending our computational resource only on active rays in the scene.

*performance analysis here*

### Part 3.2: Material sorting

*performance analysis here*

### Part 3.3: Bounding Volume Hierarchy (BVH)

*performance analysis here*

### Part 3.4: Russian roulette path termination

*performance analysis here*

## Part 4: References & Credits
* BSDF implementation:
  * PBRT: https://pbrt.org/
  * Codebase & my own implementation from CIS 5610: Advanced Rendering
* BVH: https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/ 
* OIDN : https://github.com/RenderKit/oidn
* tinyobj : https://github.com/tinyobjloader/tinyobjloader
* D.Va OBJ model:
  * character from *Overwatch*, Blizzard Entertainment
  * model from https://www.cgtrader.com/3d-print-models/miniatures/figurines/dva-and-meka
