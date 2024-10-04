CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Nadine Adnane
  * [LinkedIn](https://www.linkedin.com/in/nadnane/)
* Tested on my personal laptop (ASUS ROG Zephyrus M16):
* **OS:** Windows 11
* **Processor:** 12th Gen Intel(R) Core(TM) i9-12900H, 2500 Mhz, 14 Core(s), 20 Logical Processor(s) 
* **GPU:** NVIDIA GeForce RTX 3070 Ti Laptop GPU

Note: I am using a late day on this assignment to extend the code deadline.

### Summary

In this project, I implemented a CUDA-based path tracer capable of rendering globally-illuminated images at a fast pace.


### Part 1 - Core Features
- Shading Kernel with BSDF evaluation for:
  - Ideal diffuse surfaces
  - Perfect specular-reflective (mirrored) surfaces
  - Imperfect specular surfaces
- Path continuation/termination using Stream Compaction
- Make rays/pathSegments/intersections contiguous in memory by material type before shading
- Stochastic sampled anti-aliasing

### Part 2 - Customization! Let's have some fun!~ :D

Total needed: 10 points

1. Refraction (e.g. glass/water) (2pts)
2. Arbitrary mesh loading and rendering with toggleable bounding volume intersection culling (OBJ = 2pts, glTF = 4pts)
3. Texture mapping and bump mapping (5 pts, 6 pts if you complete arbitrary mesh loading)

### Results

## Debug Images

## Bloopers! :D

## Extra Feature Analysis

# Feature 1
- Overview
- Before/After images
- Performance impact
- GPU version vs Hypothetical CPU version
- Future Optimizations

# Feature 2
- Overview
- Before/After images
- Performance impact
- GPU version vs Hypothetical CPU version
- Future Optimizations

# Feature 3
- Overview
- Before/After images
- Performance impact
- GPU version vs Hypothetical CPU version
- Future Optimizations

## Overall Analysis

- Stream Compaction - Single iteration plot & analysis
- Stream Compaction - Open vs. Closed scene analysis
- Optimizations that target specific kernels (?)

### References & Credits
[My pathtracer from CIS-5610 Advanced Rendering](https://github.com/CIS-4610-2023/homework-05-full-lighting-and-environment-maps-nadnane/tree/main)