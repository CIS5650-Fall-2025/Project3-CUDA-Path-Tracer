CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Matt Schwartz
* Tested on: Windows 10 22H2, Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz, NVIDIA GeForce RTX 2060

<p align="center">
  <img src="img/IronMan_HeadShot_300spp.png" alt="Feature photo - ironman with environment mapping">
</p>

# Background

Welcome to my CUDA-based path tracer! What is a path tracer, I hear you asking? In its simplest form, it's a program that follows the paths of light rays as they bounce around a scene - absorbing into materials, scattering off others, refracting in yet others, etc. - and colors the pixels on your screen according to what those rays intersect and how they interact with those materials. If you follow physically-based rules for how those light rays interact with objects, you can get very realistic looking scenes!

One important note about path tracers - it's impractical to follow every light ray in existence! We really only care about the ones that make it to our "eye" / "camera"; those are the ones we see. To find those rays, the ones we see, we actually operate in reverse! We "shoot" rays out in all directions from the camera, and follow their paths until they terminate at a light source (or until we say "enough is enough!"). I like to think of it like echolocation, but with light.

That's enough context, for now - let's take a tour through some path-traced images and the features implemented in this project.

# Features
## Visual Features

### Diffuse, reflective, refractive, and emissive surfaces!

One of the marvelous abilities of path tracers is how easy it is to model a wide variety of different materials. Simply changing the distribution of paths a light ray bouncing off the material takes will take a material from metallic, to plastic, to glassy, and more.

The simplest type of material is a "diffuse" one. In terms of light scattering, a diffuse materially is equally likely to scatter incoming light in any outgoing direction (*simplification). In more common terms, we might call such a material "matte". Here's what it looks like:

</br>
</br>

On the other hand, if incoming light scatters off a surface at the same angle it came in at, we get a reflective surface:

<div align="center">
  <img src="img/reflection.png" alt="Blooper of texture mapping">
</div>

</br>
</br>

Light isn't limited to absorbing and scattering, however. It can also *refract* - or bend through materials that *transmit* light through their volumes. For example - glass, plastic, crystals, etc. The physical quantity that determines how much light bends in these media is called the index of refraction; by changing this value, we can easily render a variety of transmissive media!   

<div align="center">
  <img src="img/Refraction.png" alt="Blooper of texture mapping">
</div>

</br>
</br>

And, finally (for this project at least - there's a lot more you can do!), we have emission. Light sources! These materials emit light, and when rays hit them, they contribute to those rays' intensities and constitute the end of their journeys.

### Depth of field

<div align="center">
  <img src="img/depthoffield.png" alt="Blooper of texture mapping">
</div>

### Stochastic antialising

### Denoising

### Arbitrary mesh loading and intersection

<div align="center">
  <img src="img/modelloading.png" alt="Blooper of texture mapping">
</div>

### Texture mapping and normal mapping

<div align="center">
  <img src="img/texturemapping.png" alt="Blooper of texture mapping">
</div>

### Environment mapping

## Performance features 

### Stream compaction for early ray termination

### Material sorting

### Russian roulette termination

### Bounding Volume Hierarchy (BVH)
<div align="center">
  <img src="img/BVH.png" alt="BVH by Sebastian Lague" style="width: 600px">
</div>

# Bloopers

<div align="center">
  <div style="max-width: 1000px"><em> Texture mapping was, perhaps surprisingly, the most difficult part of this project for me. Mismatched euler angle orders between GLM and our utility functions, neglecting to transform normals appropriately, and so much more. Here's my sad, mis-textured spaceship: </em></div>
  <img src="img/texture_mapping_blooper.webp" alt="Blooper of texture mapping">
</div>

</br>
</br>


<div align="center">
  <div style="max-width: 1000px"><em> The sphere was actually behind the Ironman model in 3D space! When doing intersection tests, I had neglected to transform the intersection distance back into world space, and so objects with non-uniform scale were behaving quite oddly! </em></div>
  <img src="img/Clipping_Blooper.png" alt="Blooper of sphere clipping with ironman">
</div>

# References / Licenses

TODO