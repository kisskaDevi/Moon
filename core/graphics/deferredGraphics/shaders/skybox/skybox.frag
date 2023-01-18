#version 450

layout(set = 1, binding = 1)	uniform samplerCube samplerCubeMap;

layout(location = 0)	in vec3 inUVW;
layout(location = 1)	in float depth;
layout(location = 2)	in vec4 constColor;
layout(location = 3)	in vec4 colorFactor;

layout(location = 0) out vec4 outBaseColor;

void main()
{
    outBaseColor = colorFactor * texture(samplerCubeMap, inUVW) + constColor;
}
