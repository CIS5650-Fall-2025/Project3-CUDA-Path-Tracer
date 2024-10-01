**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 3 - CUDA Path Tracer**

* Alan Lee
  * [LinkedIn](https://www.linkedin.com/in/soohyun-alan-lee/)
* Tested on: Windows 10, AMD Ryzen 5 5600X 6-Core Processor @ 3.70GHz, 32GB RAM, NVIDIA GeForce RTX 3070 Ti (Personal Computer)

## CUDA Path Tracer

This project is a path tracer designed and implemented using C++, OpenGL, and CUDA.

The path tracer currently supports the following features:
- Path termination based on stream compaction and russian roulette
- Stochastic sampled antialiasing and ray generation using stratified sampling
- Physically-based depth-of-field
- Environment mapping
- Open Image AI Denoiser

- Lambertian (ideal diffuse), metal (perfect specular with roughness parameter), dielectric (refractive), and emissive materials
- OBJ loading with texture mapping and bump mapping
- Hierarchical spatial data structure (bounding volume hierarchy)

- Restartable path tracing

## Running the Code

You should follow the regular setup guide as described in [Project 0](https://github.com/CIS5650-Fall-2024/Project0-Getting-Started/blob/main/INSTRUCTION.md#part-21-project-instructions---cuda).

## Controls