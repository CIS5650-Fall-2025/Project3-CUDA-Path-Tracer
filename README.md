CUDA Path Tracer
================

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 0**

* Manvi Agarwal
  * [linkedIn](https://www.linkedin.com/in/manviagarwal27/)
* Tested on: Windows 11, AMD Ryzen 5 7640HS @ 4.30GHz 16GB, GeForce RTX 4060 8GB(personal)

**Features Implemented**


1. Shading kernel with BSDF evaluation for ideal specular and diffuse surfaces
2. Stream Compaction used for optimizing path termination
3. Path Segments sorting by material Id in order to reduce warp divergence
4. Refraction surface using frensel effects using Schlick approximation
5. Stochastic sampled Anti-aliasing 
6. Depth of field by jittering camera position given defocus angle and focal length
7. Direct point and direct area lighting along with simple path integrator
