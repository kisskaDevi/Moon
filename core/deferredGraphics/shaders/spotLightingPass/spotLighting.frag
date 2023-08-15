#version 450
#define MANUAL_SRGB 1
#define pi 3.141592653589793f

#include "metods/spotLight.frag"

layout (constant_id = 0) const bool enbaleShadow = false;
layout (constant_id = 1) const bool enableScattering = false;

layout(location = 0)	in vec4 eyePosition;
layout(location = 1)	in vec4 glPosition;
layout(location = 2)	in mat4 projview;

layout(input_attachment_index = 0, binding = 0) uniform subpassInput inPositionTexture;
layout(input_attachment_index = 1, binding = 1) uniform subpassInput inNormalTexture;
layout(input_attachment_index = 2, binding = 2) uniform subpassInput inBaseColorTexture;
layout(input_attachment_index = 3, binding = 3) uniform subpassInput inEmissiveTexture;
layout(input_attachment_index = 4, binding = 4) uniform subpassInput inDepthTexture;

layout(set = 2, binding = 0) uniform sampler2D shadowMap;
layout(set = 2, binding = 1) uniform sampler2D lightTexture;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outScattering;
layout(location = 2) out vec4 outBloom;

#include "metods/scattering.frag"

void main()
{
    vec4 position = subpassLoad(inPositionTexture);
    vec4 normal = subpassLoad(inNormalTexture);
    vec4 baseColorTexture = subpassLoad(inBaseColorTexture);
    vec4 emissiveTexture = subpassLoad(inEmissiveTexture);
    float depthMap = subpassLoad(inDepthTexture).r;
    float ao = emissiveTexture.a;

    outColor = ao * calcLight(position, normal, baseColorTexture, eyePosition, shadowMap, lightTexture, enbaleShadow);
    outBloom = SRGBtoLINEAR(emissiveTexture) + (checkBrightness(outColor) ? outColor : vec4(0.0f));
    
    outScattering = enableScattering ? (enbaleShadow 
        ? LightScattering(
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
            light.prop.x)       // type
        : LightScattering(
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
            depthMap, 
            light.prop.z,       // lightDropFactor
            light.prop.x)       // type
    ) : vec4(0.0f);
}
