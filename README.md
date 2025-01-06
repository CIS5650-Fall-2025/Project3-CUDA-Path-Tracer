
<h1 align="center"> CUDA Path Tracer </h1>

<small><h5 align="center">University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3</h5></small>

<!-----
<h4 align="center">Meet the Dev</h4>

| <p align="center"><br><img src="img/nadine.png" width=100><br></p> | <p><br><i> Nadine Adnane </i><br></p> [LinkedIn](https://www.linkedin.com/in/nadnane/) |
|------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|-->

 
---

<p align="center"><img src="img/denoise%20pretty.png" width=500>
</p>

## Summary

In this project, I implemented a CUDA-based path tracer capable of rendering globally-illuminated images at a fast pace thanks to the use of GPU hardware. There are several features, including support for rendering a variety of materials, loading custom scenes written in JSON format, denoising, and more.  
  
This path tracer includes a shading kernel with BSDF evaluation for a variety of materials. BSDF stands for "Bidirectional Scattering Distribution Function", which is simply a mathematical function that describes how light scatters from a surface. 

A BSDF is a quantitative representation of how light interacts with a surface, including how it's reflected, transmitted, or absorbed. In this project, I was able to implement support for diffuse, perfectly reflective, partially reflective, refractive, and emissive materials. More detail on each of these material types is included below.

## Part 1 - Core Features

### Ideal Diffuse Surfaces
Overview

| Before | After |
|--------|-------|
| pic 1  | pic 2 |


### Perfectly Reflective Surfaces
- Overview
- Before/After images

### Partially Reflective Surfaces
- Overview
- Before/After images

### Stream Compaction
- Path continuation/termination using Stream Compaction
- Stream Compaction - Single iteration plot & analysis
- Stream Compaction - Open vs. Closed scene analysis
- Optimizations that target specific kernels (?)

### Memory-Sorted Materials
- Overview
- Before/After images

### Stochastic sampled Anti-Aliasing
- Overview
- Before/After images

## Part 2 - Customization
## Russian Roulette
In path tracing, "Russian roulette" is an optimization technique used to efficiently terminate a ray's path. This is done by probabilistically deciding whether to continue tracing the ray based on its "throughput"—a value representing how much light it contributes to the final image. If the ray's contribution is likely to be minimal, the algorithm may stop tracing it to save computational resources. To maintain accuracy, the final result is adjusted to account for the probability of terminating the ray, ensuring the rendered image remains unbiased.

- Before/After images
- Performance impact
- GPU version vs Hypothetical CPU version
- Future Optimizations

## Refractive Materials
- Overview
- Before/After images
- Performance impact
- GPU version vs Hypothetical CPU version
- Future Optimizations

## Depth of Field
- Overview
- Before/After images
- Performance impact
- GPU version vs Hypothetical CPU version
- Future Optimizations

## Load OBJ
- Overview
- Before/After images
- Performance impact
- GPU version vs Hypothetical CPU version
- Future Optimizations

## Dynamic JSON Loading && Toggleable GUI Options
I added this feature mainly for fun, but also for convenience from the user's perspective. The original project base code allows for JSON scene loading via arguments passed to the main function. However, this meant that every time I wanted to load a different scene, I needed to exit the program and modify my Visual Studio project settings to point to the new .json file path every time.  

To make the process easier and to add an interactive element to my path tracer, I added an ImGUIFileDialog to the GUI with an "Open JSON File" button which allows for scene loading dynamically from the project "scenes" folder.  

One challenge with this feature was figuring out how to get the scene to refresh when a new file was loaded! I encountered various issues where the last rendered scene was still present in the background, the program would crash due to memory issues, or the new scene not showing up at all. I think I have a better understanding now of which dependencies needed to be cleared as the load function works seamlessly now :)

<p align="center"><img src="img/gui.png" width=500>
</p>

I also added a toggle button for each of the toggleable features, including:
* Russian Roulette
* Material Sorting
* Antialiasing
* DOF
* Denoising

In the future, the GUI could be further improved with additional options to customize the scene. It would be very cool to have a scene editor so the user could add objects, change material types, or change colors & sizes on the fly!

## Bloopers! :D

![uses rng instead of makeSeededRandomEngine](img/uses%20rng%20instead%20of%20makeSeededRandomEngine.png)

![Screenshot 2025-01-05 155737](img/Screenshot%202025-01-05%20155737.png)


![cool_scene.2025 01 05_20 08 51z.1456samp](img/cool_scene.2025-01-05_20-08-51z.1456samp.png)

Something bad is happening... I don't think this is denoising 
![denoiseBlooper](img/blooper.png)


![what](img/what.png)


## References & Helpful Resources
* [My pathtracer from CIS-5610 Advanced Rendering](https://github.com/CIS-4610-2023/homework-05-full-lighting-and-environment-maps-nadnane/tree/main) 
* [Thrust Library Documentation](https://nvidia.github.io/cccl/thrust/api/function_group__sorting_1ga667333ee2e067bb7da3fb1b8ab6d348c.html) 
* [OIDN Documentation](https://github.com/RenderKit/oidn)
* [DOF and Antialiasing](https://paulbourke.net/miscellaneous/raytracing/)