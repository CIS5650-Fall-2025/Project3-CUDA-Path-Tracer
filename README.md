CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.

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
      - [ ] env light
    - [x] Integrate these two sample strategies
    - [ ] MIS env map
    - [ ] MIS based on luminance of light
- [ ] DOF
- [ ] Denoising
    - [ ] OpenImage Denoiser built [OpenImage](https://www.openimagedenoise.org/)
        - CPU only for now
    - [ ] Integrate it into project

### LOG
- 2024.9.27
  