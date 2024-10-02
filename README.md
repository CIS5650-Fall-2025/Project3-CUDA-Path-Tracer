**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 3 - CUDA Path Tracer**

* Alan Lee
  * [LinkedIn](https://www.linkedin.com/in/soohyun-alan-lee/)
* Tested on: Windows 10, AMD Ryzen 5 5600X 6-Core Processor @ 3.70GHz, 32GB RAM, NVIDIA GeForce RTX 3070 Ti (Personal Computer)

## CUDA Path Tracer

This project is a path tracer designed and implemented using C++, OpenGL, and CUDA.

The path tracer currently supports the following features:
* Path termination based on stream compaction and russian roulette
* Stochastic sampled antialiasing and ray generation using stratified sampling
* Physically-based depth-of-field
* Environment mapping
* Open Image AI Denoiser  
---
* Loading and reading a json scene description format
* Lambertian (ideal diffuse), metal (perfect specular with roughness parameter), dielectric (refractive), and emissive materials
* OBJ loading with texture mapping and bump mapping
* Support for loading jpg, png, hdr, and exr image formats and saving rendered images as png files
* Hierarchical spatial data structure (bounding volume hierarchy)  
---
* Restartable path tracing

## Contents

* `src/` C++/CUDA source files.
* `scenes/` Example scene description JSON files.
* `img/` Renders of example scene description files.
* `renderstates/` Paused renders are stored at and read from this directory.
* `external/` Includes and static libraries for 3rd party libraries.

## Running the Code

You should follow the regular setup guide as described in [Project 0](https://github.com/CIS5650-Fall-2024/Project0-Getting-Started/blob/main/INSTRUCTION.md#part-21-project-instructions---cuda).

The main function requires either a scene description file or a render state directory. For example, the user may call the program with one as an argument: `scenes/sphere.json` for scene description files and `renderstates/` for the render state directory. (In Visual Studio, `../scenes/sphere.json` and `../rednerstates/`)

If you are using Visual Studio, you can set this in the `Debugging > Command Arguments` section in the `Project Properties`. Make sure you get the path right - read the console for errors.

### Controls

Click, hold, and move `LMB` in any direction to change the viewing direction.\
Click, hold, and move `RMB` up and down to zoom in and out.\
Click, hold, and move `MMB` in any direction to move the LOOKAT point in the scene's X/Z plane.\
Press `Space` to re-center the camera at the original scene lookAt point.

Press `S` to save current image.\
Press `Escape` to save current image and exit application.\
Press `P` to save current state of rendering to be resumed at a later time.

## Analysis


For each extra feature, you must provide the following analysis:

* Overview write-up of the feature along with before/after images.
* Performance impact of the feature.
* If you did something to accelerate the feature, what did you do and why?
* Compare your GPU version of the feature to a HYPOTHETICAL CPU version (you don't have to implement it!). Does it benefit or suffer from being implemented on the GPU?
* How might this feature be optimized beyond your current implementation?


* Stream compaction helps most after a few bounces. Print and plot the effects of stream compaction within a single iteration (i.e. the number of unterminated rays after each bounce) and evaluate the benefits you get from stream compaction.
* Compare scenes which are open (like the given cornell box) and closed (i.e. no light can escape the scene). Again, compare the performance effects of stream compaction! Remember, stream compaction only affects rays which terminate, so what might you expect?
* For optimizations that target specific kernels, we recommend using stacked bar graphs to convey total execution time and improvements in individual kernels. For example:

  ![Clearly the Macchiato is optimal.](img/stacked_bar_graph.png)

  Timings from NSight should be very useful for generating these kinds of charts.

## Attributions and Credits

### Resources
Bump map images : https://gamemaker.io/en/blog/using-normal-maps-to-light-your-2d-game

### Conceptual and Code
Bump mapping : https://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping/ \
BVH : https://github.com/CMU-Graphics/Scotty3D \
OIDN : https://github.com/RenderKit/oidn \
stb image : https://github.com/nothings/stb \
tinyexr : https://github.com/syoyo/tinyexr \
tinyobj : https://github.com/tinyobjloader/tinyobjloader \