#version 450

#include "../__methods__/defines.glsl"
#include "../__methods__/colorFunctions.glsl"
#include "../__methods__/geometricFunctions.glsl"

layout(push_constant) uniform PC {
    float minAmbientFactor;
} pc;

layout(input_attachment_index = 0, binding = 0) uniform subpassInput inPositionTexture;
layout(input_attachment_index = 1, binding = 1) uniform subpassInput inNormalTexture;
layout(input_attachment_index = 2, binding = 2) uniform subpassInput inBaseColorTexture;
layout(input_attachment_index = 3, binding = 3) uniform subpassInput inEmissiveTexture;
layout(input_attachment_index = 4, binding = 4) uniform subpassInput inDepthTexture;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outBlur;
layout(location = 2) out vec4 outBloom;

vec4 position;
vec4 normal;
vec4 baseColorTexture;
vec4 emissiveTexture;

vec4 ambient() {
    vec4 baseColor = SRGBtoLINEAR(baseColorTexture);
    vec3 diffuseColor = pc.minAmbientFactor * baseColor.rgb;
    return vec4(diffuseColor.xyz, baseColorTexture.a);
}

void main() {
    position = subpassLoad(inPositionTexture);
    normal = subpassLoad(inNormalTexture);
    baseColorTexture = subpassLoad(inBaseColorTexture);
    emissiveTexture = subpassLoad(inEmissiveTexture);

    outColor = SRGBtoLINEAR(emissiveTexture) + (checkZeroNormal(normal.xyz) ? SRGBtoLINEAR(baseColorTexture) : ambient());
    outBloom = SRGBtoLINEAR(emissiveTexture) + (checkBrightness(outColor) ? outColor : vec4(0.0, 0.0, 0.0, 1.0));
    outBlur = vec4(0.0, 0.0, 0.0, 0.0);
}
