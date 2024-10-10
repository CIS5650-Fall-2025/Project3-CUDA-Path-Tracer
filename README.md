# CUDA Path Tracer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

- Jinxiang Wang
- Tested on: Windows 11, AMD Ryzen 9 8945HS w/ Radeon 780M Graphics 4.00 GHz 32GB, RTX 4070 Laptop 8 GB

## Results Preview:

![](results/Cover/CoverDenoisedMIS.png)

<p align="center">A large mineway castle shaded with one Disney BRDF material (6,490,766 triangles, 300 spp, 6.1 fps)</p>

![](results/RefractDragon.png)

<p align="center">A stanford dragon shaded with refract material</p>

## Features Implemented:

1. Bounding Volume Hierarchy (HLBVH)
2. Disney DRDF model (not fully correct in some cases)
3. GUI for dynamic material modification
4. Muiti-Importance Sampling
5. ACES Tone Mapping
6. TinyObj mesh loading
7. Stocastic Sampled Anti-Aliasing

# Bounding Volume Hierarchy

The base code shoots rays out from the camera and compute intersections with all objects in the scene.

This works fine when we can implicitly define intersecting method for each geometry (just like SDF). But after mesh loading feature is implemented, the way of light-scene intersection is changed from calculating
intersection with implicit geometries to calculating it with all triangles!

This gives an extremely low performance when model with many faces is loaded:
| < 200 faces (~60 fps) | ~6000 faces (< 10 fps) |
| -------------------------------- | ------------------------------- |
| ![](results/BVH/simplescene.png) | ![](results/BVH/marioScene.png) |

To effectively reduce the amout of intersection computation, we could use BVH, Bounding Volume Hierarchy, which construct
a tree-like structure to store scene primitives.

The efficiency of BVH depends on how we build the tree. There are many ways to segment triangls, for this project, I used HLBVH, which is a combination of Surface Area Heuristic (SAH) and morton code based Linear BVH. For more reference, check [PBRT 4.3 Bounding Volume Hierarchy](https://pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies).

And this gives a very good speed up

| Before BVH (< 10 fps)           | After BVH (~30 fps)           |
| ------------------------------- | ----------------------------- |
| ![](results/BVH/marioScene.png) | ![](results/BVH/AfterBVH.png) |

We can go even further with a stanford dragon (2,349,078 triangles)

| Dragon ( 15 fps)            |
| --------------------------- |
| ![](results/BVH/Dragon.png) |

Visualizer:
| Wahoo | Dragon | Mineway Castle |
| ------------------------------- | ----------------------------- | --------------------------- |
| 5117 triangles | 2,349,078 triangles | 6,490,766 triangles |
| ![](results/BVH/BVH.png) | ![](results/BVH/DragonBVH.png) | ![](results/BVH/CoverBVH.png) |

# Disney BRDF Model

This is a robust and art-oriented material model that allows for interpolation between different types of material based on pbr parameters. The model implemented in this project referenced SIGGRAPH 2012 [Physically Based Shading at Disney](https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf) by Disney and their public repo [brdf](https://github.com/wdas/brdf/tree/main)
![](results/DisneyBRDF/presentation.png)

The input parameters are given by:

color baseColor .82 .67 .16  
float metallic 0 1 0  
float subsurface 0 1 0  
float specular 0 1 .5  
float roughness 0 1 .5  
float specularTint 0 1 0  
float anisotropic 0 1 0  
float sheen 0 1 0  
float sheenTint 0 1 .5  
float clearcoat 0 1 0  
float clearcoatGloss 0 1 1

| JSON Input                            |
| ------------------------------------- |
| ![](results/DisneyBRDF/jsonInput.png) |

GUI allowing for dynamically changing parameters:

| GUI                             |
| ------------------------------- |
| ![](results/DisneyBRDF/GUI.png) |

A brief demo illustrates the usage:

<video width="834" height="469" controls>
  <source src="results/DisneyBRDF/DisneyBRDF.mp4" type="video/mp4">
  Your browser does not support the video.
</video>
