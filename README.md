CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Xinran Tao
  - [LinkedIn](https://www.linkedin.com/in/xinran-tao/), [Personal Website](https://www.xinrantao.com/), [GitHub](https://github.com/theBoilingPoint).
* Tested on: 
  - Ubuntu 22.04, i7-11700K @ 3.60GHz × 16, RAM 32GB, GeForce RTX 3080 Ti 12GB (Personal)

# Base Credit
## Basic BRDF
OMG glm's reflect!!!!!
### Mirror Material Shows White with Shallow Depths
## Stream Compaction
## Sorting Paths by Material
## Antialiasing

# Extra Credit
- Visual Improvements
  - BRDFs
    - (2) Dielectric Material
    - (?) Microfacet Material
  - (2) Depth of Field
  - (6) Texture Loading with Mesh Loading
    - Can read textures from file and have a asic procedural texture (checkerboard in pathtracer.cu)
- Mesh Loading 
  - (2) obj
  - (4) GLTF
- Performance
  - (1) Raussian Roulette
  - (?) Buggy BVH
    - I am able to construct the nodes of the BVH an traverse it, but somehow the ray misses the triangle in basically all leaf nodes. I don't know why.


