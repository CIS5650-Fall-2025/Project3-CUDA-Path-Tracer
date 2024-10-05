CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Zhiyi Zhou
* Tested on: Windows 11, i9-13900H @ 2.6GHz 64GB, RTX4060 Laptop

### Features
- OBJ Mesh Loading
- BVH(MTBVH) Acceleration
- Stochastic Sampled Anti-Aliasing
- Physically-Based Materials
- Texture Mapping & Normal Mapping
- Environment Mapping
- Multiple Importance Sampling of different kind of light(sphere, cube, mesh and environment light)

![](img2024/glassbunny2.png)
![](img2024/camera.png)

| Direct Light sample(20spp)   | BSDF Sample (20spp)      | MIS(20spp)                  |
| :-----------------------:    | :----------------------: | --------------------------- |
| ![](./img2024/Direct20.png)  | ![](./img2024/BSDF20.png)| ![](./img2024/MIS20.png)    |

| Direct Light sample(2000spp)   | BSDF Sample (2000spp)      | MIS(2000spp)                  |
| :-----------------------:    | :----------------------: | --------------------------- |
| ![](./img2024/Direct2000.png)  | ![](./img2024/BSDF2000.png)| ![](./img2024/MIS2000.png)    |

TODO:
- [ ] MIS
    - [x] BSDF sample(Lambertian, Dielectric, Microfacet, Metallic)
    - [ ] Light sample(light of different shape: sphere, cube, plane, triangles)
      - [x] sphere light
      - [x] cube light
      - [x] obj(triangles) light
      - [x] env light
    - [x] Integrate these two sample strategies
    - [x] MIS env map
- [ ] DOF
- [ ] Denoising
    - [ ] OpenImage Denoiser built [OpenImage](https://www.openimagedenoise.org/)
        - CPU only for now
    - [ ] Integrate it into project