# Deferred Graphics

## Optimization

I use a single vkCmdDraw for each light sources, in which every single pass draw inside surface of piramid, corresponded to projection and view matrixes of light source. Fragment shader works only with pyramid fragments of this light point. We do not shade points outside of light volume by this way. Results of every light pass is blended in attachment using function of maximum. We have image with lighted fragments of pyramids. For normal result I do final pass of ambient light. It fills fragments which outs of light pyramids.
Eventually we have same image of deferred render, but calculations are more fast. Also differentiation of light sources calculations lets using various pipelines and shaders for every light source. It lets us render variose effects without performance drop. In fact performance depends on count of shaded fragments and do not depend on light sources count.