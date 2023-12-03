#version 450

#include "../__methods__/defines.glsl"
#include "../__methods__/colorFunctions.glsl"

#include "spotLightingBase.frag"

layout(location = 0) in vec4 eyePosition;

layout(input_attachment_index = 0, binding = 0) uniform subpassInput inPositionTexture;
layout(input_attachment_index = 1, binding = 1) uniform subpassInput inNormalTexture;
layout(input_attachment_index = 2, binding = 2) uniform subpassInput inBaseColorTexture;
layout(input_attachment_index = 3, binding = 3) uniform subpassInput inDepthTexture;

layout(set = 2, binding = 0) uniform sampler2D shadowMap;
layout(set = 2, binding = 1) uniform sampler2D lightTexture;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outBlur;
layout(location = 2) out vec4 outBloom;

void main() {
    vec4 position = subpassLoad(inPositionTexture);
    vec4 normal = subpassLoad(inNormalTexture);
    vec4 baseColorTexture = subpassLoad(inBaseColorTexture);

    vec4 emissiveAndAO = decodeFromFloat(normal.a);
    vec4 emissiveTexture = vec4(emissiveAndAO.xyz, 1.0f);
    float ao = emissiveAndAO.a;

    outColor = ao * calcLight(position, normal, baseColorTexture, eyePosition, shadowMap, lightTexture);
    outBloom = SRGBtoLINEAR(emissiveTexture) + (checkBrightness(outColor) ? outColor : vec4(0.0));
    outBlur = vec4(0.0, 0.0, 0.0, 1.0);
}
