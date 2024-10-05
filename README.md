CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Zhen Ren
  * https://www.linkedin.com/in/zhen-ren-837089208/
* Tested on: Windows 11, i9-13900H @ 2.60 GHz 16GB, RTX 4070 Laptop 8GB (Self laptop)
![](./img/mesa5.png)

## User Instructions
### Hot Keys
| Keys| Usage               |
|-----|---------------------|
| ESC | Quit and save image |
| P   | Save image          |
| Z   | Zoom in             |
| X   | Zoom out            |
| W   | Moving up           |
| S   | Moving down         |
| F   | Auto focus          |

### Auto Focus
Auto focus is a very convenient helper to let you focus on any object you want. Hovering your mouse over the object you want to focus and then press **F**. This object will then be focused, i.e. the focal point is set to the object's surface. Also works for the tilt shift camera.

### Mouse control
The same as default setting. Left to rotate the camera, right to zoom the camera. Hold middle button to snap around the camera.

### ImGui Settings
ImGui is mainly used to control tilt shift camera. You can control the aperture size, lens shift and also the direction of focal plane.
You can also toggle ray compaction and material sort here.

## Gallery
| Mesa - shot using thin lens camera (1920*1080 100spp aperture0.5)|
| :------------------------------------: |
|![](./img/mesa4.png)|

| Mesa - shot using tilt shift camera (1920*1080 100spp aperture0.5)|
| :------------------------------------: |
|![](./img/mesa1.png)|



<center><b>Vokselia Spawn (1920*1080 100spp 0.5aperture size)</b></center>

|With tile shift camera|Normal thin lens camera|
| :-: | :-: |
| ![](./img/vokselia1.png)| ![](./img/vokselia2.png) |

Using the same aperture size, tilt shift camera can generate more focused images. So, tilt shift camera can achieve a miniature faking effect.

| SpongeBob - shot using tilt shift camera (1920*1080 100spp)|
| :------------------------------------: |
|![](./img/spongebob1.png)|

We can also focus on arbitary plane, like a set of buildings on one side of the street.

| Dewmire Castle (1080*1080 100spp)|
| :------------------------------------: |
|![](./img/dewmire_castle.png)|

| Mando Helmet (1000*1000 76spp)|
| :------------------------------------: |
|![](./img/gltf1.png)|

| White Girl Robot (1080*1080 500spp)|
| :------------------------------------: |
|![](./img/gltf2.png)|

| Glass Dragon (1080*1080 1000spp)|
| :------------------------------------: |
|![](./img/fraction1.png)|

| Glass Bunny (1080*1080 1000spp)|
| :------------------------------------: |
|![](./img/fraction2.png)|

<center><b>Rungholt (1920*1080 10spp)</b></center>

|With MIS|Naive sampling|
| :-: | :-: |
| ![](./img/rungholt1.png)| ![](./img/rungholt2.png) |