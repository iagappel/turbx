---
layout: default
title: Home
nav_order: 1
permalink: /
---

# **turbx documentation**

`turbx` is available on [Github](https://github.com/iagappel/turbx)

```
git clone git@github.com:iagappel/turbx.git
```

## Introduction

`turbx` is a post-processing library for the analysis of turbulent flow datasets.
<!--- -->
It takes advantage of collective IO and parallelized data processing to achieve relatively high performance in a `python3` framework, made possible primarily through the `h5py` and `mpi4py` python packages.
<!--- -->
This allows for high performance as well as flexible usage, high maintainability and a low barrier to entry for collaborating users.

## Visualization

The main `turbx` data classes have `make_xdmf()` functions which output `XDMF` data descriptor referring to their `HDF5` data container. This allows for visualization in `Paraview` among other post-processing software.

<!---
This is a comment
-->

<!---
<iframe
width="560" height="315"
src="https://www.youtube.com/embed/DR-1spwaYPw?vq=hd1440" 
frameborder="0"
allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" 
allowfullscreen></iframe>
-->

<style>
.yt {
  position: relative;
  display: block;
  width: 100%; /* width of iframe wrapper */
  height: 0;
  margin: auto;
  padding: 0% 0% 56.25%; /* 16:9 ratio */
  overflow: hidden;
}
.yt iframe {
  position: absolute;
  top: 0; bottom: 0; left: 0;
  width: 100%;
  height: 100%;
  border: 0;
}
</style>

<!---
============================================================
-->

Here is a visualization of particles tracked through a 3D, unsteady turbulent flow field using `turbx`.

<div class="yt">
  <iframe
  width="560" height="315"
  src="https://www.youtube.com/embed/DR-1spwaYPw?vq=hd2160" 
  frameborder="0"
  allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" 
  allowfullscreen></iframe>
</div>

<!---
... more examples needed! ...
-->

