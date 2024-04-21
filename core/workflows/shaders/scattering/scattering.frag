#version 450

#include "../__methods__/defines.glsl"
#include "scatteringBase.glsl"

layout(location = 0)	in vec4 eyePosition;
layout(location = 1)	in vec4 glPosition;
layout(location = 2)	in mat4 projview;

layout(set = 0, binding = 1) uniform sampler2D inDepthTexture;

layout(set = 2, binding = 0) uniform LightBufferObject
{
    mat4 proj;
    mat4 view;
    mat4 projView;
    vec4 position;
    vec4 color;
    vec4 prop;
} light;

layout(set = 3, binding = 0) uniform sampler2D lightTexture;
layout(set = 1, binding = 0) uniform sampler2D shadowMap;

layout (push_constant) uniform PC
{
    int width;
    int height;
}pc;

layout(location = 0) out vec4 outScattering;

void main()
{
    float depthMap = texture(inDepthTexture, vec2(gl_FragCoord.x / pc.width, gl_FragCoord.y / pc.height)).r;

    outScattering = LightScattering(
            50, 
            light.view, 
            light.proj, 
            light.projView, 
            light.position, 
            light.color, 
            projview, 
            eyePosition, 
            glPosition, 
            lightTexture, 
            shadowMap, 
            depthMap, 
            light.prop.z,       // lightDropFactor
            light.prop.x);      // type
}
