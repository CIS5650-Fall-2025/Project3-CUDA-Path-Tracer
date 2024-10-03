CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Daniel Gerhardt
  * https://www.linkedin.com/in/daniel-gerhardt-bb012722b/
* Tested on: Windows 23H2, AMD Ryzen 9 7940HS @ 4GHz 32GB, RTX 4070 8 GB (Personal Laptop)

### CUDA Path Tracer

## Table of Contents:
* Description
  * Controls
  * Toggling Features
  * Detailed Feature Overview
  * Scene File Format
* Performance Analysis
	* Stream Compaction
	* Material Sorting
	* Textures and Bump Maps
	* Image Denoising
	* Environment Map
* Issues
  * Challenging Bugs
  * Bloopers
  * TODO
* Acknowledgements and Resources

## Description

Path tracing is the process of shooting a ray out of each pixel of the screen from the "camera", and collecting color by bouncing the ray around throughout the 3D scene. This project was done using CUDA to run the compute-heavy processes in parallel on the GPU. The path tracer was completed with the following features(* = core feature):
1. Ideal diffuse and specular surfaces*
2. Dielectric materials like glass with refraction
3. Stream compaction for terminating non-contributing paths*
4. Sorting intersections by material type*
5. Stochastic sampled antialising by jittering rays within each pixel*
6. Mesh loading with OBJ files
7. Bounding Volume Hierarchy(BVH)
8. Environment mapping
9. Texture and bump mapping with optional procedural texture
10. Real time and final render denoising with Intel Open Image Denoise
See the detailed feature overview below to read more about these features.

### Controls
* Left mouse button - rotates the camera.
* Right mouse button on the vertical axis - zooms in or out.
* Middle mouse button - moves camera on the global X/Z plane.
* Space - re-centers the camera.
* S - saves an image to the build folder.
* Esc - saves an image and closes the program.

### Toggling Features
The following features are toggleable and can be enabled or disabled for different image and performance effects. Locate the following defines to change the status of the feature.
1. Stream compaction: USE_STREAM_COMPACTION - 1 = enabled, 0 = disabled. Disabling this will decrease performance since more paths have to be analyzed.
2. Intersection sorting by material:  USE_MATERIAL_SORTING - 1 = enabled, 0 = disabled. Disabling this will descrease performance since neighboring threads will be utilizing less contiguous memory.
3. Bounding volume hierarchy: USE_BVH - 1 = enabled, 0 = disabled. Disabling this will decrease performance when rendering OBJ files, since rays will be tested against every triangle rather than the bounding volume hierarchy.
4. Bump map: USE_BUMP_MAP - 1 = enabled, 0 = disabled. Disabling this will not use bump map textures for meshes that specify a bump map.
5. Texture from file: USE_PROCEDURAL_TEXTURE - 1 = use texture from file, 0 = use procedural texture. Enabling this will set all meshes that have a specified texture to use the procedural texture.
6. Environment map: USE_ENVIRONMENT_MAP - 1 = enabled, 0 = disabled. Disabling this shows a black background rather than the 360 degree environment view.
7. Use image denoising for real time render view: USE_OIDN_FOR_RENDER - 1 = enabled, 0 = disabled. Disabling this will increase performance but the rendered image will be more noisy.
8. Use image denoising for final image saving: USE_OIDN_FINAL_IMAGE - 1 = enabled, 0 = disabled. Disabling this will save the raw but more noisy final render.

### Detailed Feature Overview
1. Ideal diffuse and specular surfaces. These are surface types that are the most basic in path tracing. Ideal diffuse surfaces will reflect light with an equal probability in every direction. Ideal specular surfaces always reflect light in one direction, reflected about the surface normal, like a mirror. Neither of these surfaces exist perfectly in real life but they are convenient to implement in a path tracer.

On the left is a perfectly diffuse red sphere, and on the right a perfectly specular chrome sphere.
![](renders/diffuse_and_specular.png)

2. Dielectric materials. Some materials, like glass, both reflect light outwards and refract light inwards. This phenomenon causes caustics, which is focused light through a transmissive material.

Glass sphere with reflection and refraction:
![](renders/dielectric_demo.png)

3. Stream compaction for terminating non-contributing paths. Stream compaction is the process of removing elements from an array that do not meet a certain criteria. In a path tracer, this can be used to remove rays that have finished bouncing or have bounced into the outer reaches of the scene from consideration of future computation. See the performance analysis below for a detailed analysis of how this speeds up the path tracer.

ADD PERFORMANCE LINK

4. Sorting intersections by material type. In a parallel environment, multiple threads that are continguous will be slowed down by working on memory that is spread out in a random manner. Each thread will be assigned to an intersection. Within the shading stage, different memory is accessed based upon the material type, and different code is executed based on the material as well. So, sorting the intersections by material will increase the coherency of the memory and decrease the diveregence between neighboring threads. See the performance analysis below for a detailed analysis of how this speeds up the path tracer.

ADD PERFORMANCE LINK

5. Stochastic sampled antialising by jittering rays within each pixel. Antialising is smoothing out rough edges. This can be done "for free" within a path tracer without extra computation by slightly moving the ray position, which will cause the pixel to draw color from slightly different positions in the scene, effectively blurring the pixel color and smoothing out the rough edges.

Image with no antialiasing: ![](renders/no_aliasing_zoom.png)

Image with antialiasing: ![](renders/yes_aliasing_zoom.png)

6. Mesh loading with OBJ files. The OBJ format is a standardized and common way of representing complex objects. There is support for loading arbitrary OBJ files, along with their textures and bump maps. I chose to use TinyOBJ to read in the data, and then passing it to the GPU as an array of triangles.

obj image: 

7. Bounding Volume Hierarchy. A naive approach to rendering in a path tracer is to test if a ray intersects with any object in the scene by doing an intersect test with each primitive object(triangles, planes, spheres). This can be extremely slow if there are complex objects made up of many primitive objects, which is common of OBJ files that are made of triangles. A bounding volume hierarchy reduces the number of primitives that are checked against. To do this, a volume is created to enclose the triangles in the scene(in my case, the volumes are cubes). Then the volume is divided over and over, until each of the smallest volume divisions encloses one or two primitives. The ray can be checked against the larger volumes to rule out many primitives, and only has to be compared against log2(n) primitives rather than n primitives. See the performance analysis below for a detailed analysis of how this speeds up the path tracer(hint: A LOT).

Visualization

8. Environment mapping. If a ray does not hit anything in the scene, the basic technique is to make the color at that point black. This gives the viewer a sense of dread, which is generally not the goal in computer graphics. To alleviate this fear inducing void, the rays that are sent in to the void can instead have their direction mapped to a cubemap texture coordinate, and a nice environment can be created around the scene.

Scary void:

Nice environment:

9. Texture and bump mapping with optional procedural texture. Object files are often colored with textures. Additionally, a technique called bump mapping can be used to give artificial small details by varying the normals based on a texture called a bump map. To achieve this in the path tracer, the primary challenge is getting the data and indexing correctly on the GPU. To do this, I am passing a large array of colors to the GPU, along with an array of start indices and directions. The triangle primitives that are intersected with carry a texture index, and this can be used to sample the start index and dimension arrays to get a final index to sample the color array.

Textured object:

Bumpy object:

Textured bumpy object:

10. Real time and final render denoising with Intel Open Image Denoise. A big problem with path tracing is it can take a long time for the speckles in the image to be smoothed out. These specks are caused by the time it takes for a ray to be cast at each point in the scene, and it can take multiple rays at the points to provide an accurate and visually pleasing color. These speckles, called noise, can be dealt with by using a denoiser. Intel provides a deep learning based denoiser that is rather easily integrated into the path tracer. It can be used every frame to denoise the render view, or used with prefiltering on the final saved image. Prefiltering is not used for every frame because it is slow.

Denoising images:

### Scene file format
The scenes are stored in JSON files for easy parsing. There are 3 main sections.

1. Materials. Materials have unique names and a series of parameters. The first is type. Materials have the following supported types: `"Diffuse"`, `"Specular"`, `"Emitting"`, `"SpecularTransmissive"`, `"Texture"`, `"BumpMap"`, and `"EnvironmentMap"`. The other parameters depend on the type. `"Diffuse"` and `"Specular"` materials require `"RGB"`. `"Emitting"` in addition to `"RGB"` requires `"Emittance"`. `"SpecularTransmissive"` in addition to `"RGB"` also requires `"ETA"`. `"Texture"`, `"BumpMap"`, and `"EnvironmentMap"` require the `"FILE"`, as well as `"WIDTH"` and `"HEIGHT"`.

Examples:
```
"Materials":
{
	"light":{
	    "TYPE":"Emitting",
	    "RGB":[1.0, 1.0, 1.0],
	    "EMITTANCE":5.0
	},
	"diffuse_white":
	{
	    "TYPE":"Diffuse",
	    "RGB":[0.98, 0.98, 0.98]
	},
	"specular_white":
	{
	    "TYPE":"Specular",
	    "RGB":[0.98, 0.98, 0.98],
	    "ROUGHNESS":0.0
	},
	"specular_transmissive_white":
	{
	    "TYPE":"SpecularTransmissive",
	    "RGB":[0.98, 0.98, 0.98],
	    "ETA":[1.0, 1.55]
	},
	"dog_tex":
	{
	    "TYPE":"Texture",
	    "FILE":"C:/Users/danie/Desktop/School/CIS 5650/Project3/scenes/textures/wolftexture.png",
	    "WIDTH":64,
	    "HEIGHT":64
	},"dodecahedron_bump":
	{
	    "TYPE":"BumpMap",
	    "FILE":"C:/Users/danie/Desktop/School/CIS 5650/Project3/scenes/textures/154_norm.JPG",
	    "WIDTH":1024,
	    "HEIGHT":1024
	},
	"env_map":
	{
	    "TYPE":"EnvironmentMap",
	    "FILE":"C:/Users/danie/Desktop/School/CIS 5650/Project3/scenes/environmentmaps/Frozen_Waterfall_Ref.hdr",
	    "WIDTH":1600,
	    "HEIGHT":800
	}
}
```

2. Camera. The camera requires the following fields seen in this example:
```
"Camera":
{
    "RES":[800,800],
    "FOVY":45.0,
    "ITERATIONS":5000,
    "DEPTH":8,
    "FILE":"cornell",
    "EYE":[0.0,5.0,10.5],
    "LOOKAT":[0.0,5.0,0.0],
    "UP":[0.0,1.0,0.0]
}
```

3. Objects. Each object contains:
- `"TYPE"`: The type of object, such as `"cube"` or `"sphere"`.
- `"MATERIAL"`: The material assigned to the object, referencing one of the materials defined earlier.
- `"TRANS"`: An array for the translation (position) of the object.
- `"ROTAT"`: An array for the rotation of the object in degrees.
- `"SCALE"`: An array for the scale of the object.

There are additional required parameters if the `"TYPE"` is `"mesh"`.
- `"BUMPMAP"`: The bumpmap material, or `""` if there is no bumpmap.
- `"FILE"`: The OBJ file for the mesh.
- `"FILE_FOLDER"`: The folder with the OBJ file for the mesh.

Examples:
```
"Objects":
{
	{
        "TYPE":"cube",
        "MATERIAL":"diffuse_green",
        "TRANS":[5.0,5.0,0.0],
        "ROTAT":[0.0,0.0,0.0],
        "SCALE":[0.01,10.0,10.0]
    },
    {
        "TYPE":"mesh",
        "MATERIAL":"ninja_tex",
        "BUMPMAP": "ninja_bumpmap",
        "TRANS":[0.0,0.0,0.0],
        "ROTAT":[0.0,0.0,0.0],
        "SCALE":[1.0,1.0,1.0],
        "FILE":"C:/Users/danie/Desktop/School/CIS 5650/Project3/scenes/objs/ninja.obj",
        "FILE_FOLDER":"C:/Users/danie/Desktop/School/CIS 5650/Project3/scenes/objs/"
    },
    {
        "TYPE":"mesh",
        "MATERIAL":"dog_tex",
        "BUMPMAP": "",
        "TRANS":[2.0,4.0,-3.4],
        "ROTAT":[0.0,0.0,0.0],
        "SCALE":[0.2,0.2,0.2],
        "FILE":"C:/Users/danie/Desktop/School/CIS 5650/Project3/scenes/objs/wolf.obj",
        "FILE_FOLDER":"C:/Users/danie/Desktop/School/CIS 5650/Project3/scenes/objs/"
    }
}
```

## Performance Analysis

### Stream Compaction

### Material Sorting

### Bounding Volume Hierarchy

### Textures and Bump Maps

### Image Denoising

### Environment Map

## Issues

### Challenging Bugs

### Bloopers

### TODO

## Acknowledgements and Resources
 


