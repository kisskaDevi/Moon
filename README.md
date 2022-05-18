# Vulkan C++ application

<img src="./screenshots/screenshot2.PNG" width="1000px"> <img src="./screenshots/screenshot1.PNG" width="1000px">

## About

This application makes render of scene by Vulkan API and contains some modern graphics effects, which I will give below.

## Application scheme
<img src="./screenshots/scheme.png" width="1000px">

## Content

* Base render of 3D geometry and texturing.
* Spot and point light sources with shadows.
* Reneder in several attachments.
* Two pass render - first pass draw and shade scene, second pass is post processing.
* glTF models render based on [repository of Sascha Willems](https://github.com/SaschaWillems/Vulkan-glTF-PBR), redesigned for this implementation of the Vulkan application.
* Animations and linear interpolation between Animations
<img src="./screenshots/Vulkan.gif" width="1000px">

* Groups of objects, light sources, camera and groups manipulations.
* Using three frame buffers.
* MSAA.
* MIP-maps.
* Skybox

* Bloom by bliting of source image
<img src="./screenshots/screenshot3.PNG" width="1000px">

* deferred render by subpasses

<img src="./screenshots/screenshot5.PNG" width="400px"> <img src="./screenshots/screenshot6.PNG" width="400px">
<img src="./screenshots/screenshot7.PNG" width="400px"> <img src="./screenshots/screenshot8.PNG" width="400px">

* Volumetric light
<img src="./screenshots/screenshot4.PNG" width="1000px">

* Screen Space Local Reflections
<img src="./screenshots/screenshot9.PNG" width="1000px">

* Using stencil buffer for highlighting objects
<img src="./screenshots/stencil.gif" width="1000px">

* Prototype of area light (sphere sources and surfaces)
<img src="./screenshots/areaLight.gif" width="1000px">

* Store buffer, which can be readen by CPU, for example, I used it for detect object under cursor
* Simple physics
<img src="./screenshots/collisions.gif" width="1000px">

## Optimization

I use a single vkCmdDraw for each light sources, in which every single pass draw inside surface of piramid, corresponded projection and view matrixes of light source. Fragment shader works only with pyramid fragments of this light point. We do not shade points out of light volume by this way. Results of every light pass is blended in attachment using function of maximum. We have image with lighted fragments of pyramids. For normal result I do final pass of ambient light. It fills fragments which outs of light pyramids.
<img src="./screenshots/screenshot11.PNG" width="400px"> <img src="./screenshots/screenshot12.PNG" width="400px">

Eventually we have same image of deferred render, but calculations are more fast. Also differentiation of light sources calculations lets using various pipelines and shaders for every light source. It lets us render variose effects without performance drop. In fact performance depends on count of shaded fragments and do not depend on light sources count.


