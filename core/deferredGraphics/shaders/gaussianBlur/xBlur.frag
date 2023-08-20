#version 450

#include "../__methods__/defines.glsl"

layout(set = 0, binding = 0) uniform sampler2D blurSampler;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outBlur;

float weight[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

vec4 blur(sampler2D blurSampler, vec2 TexCoord) {
    vec2 texOffset = 1.0 / textureSize(blurSampler, 0);
    vec3 result = texture(blurSampler, TexCoord).rgb * weight[0];

    for(int i = 1; i < 5; ++i) {
        result += texture(blurSampler, TexCoord + vec2(texOffset.x * i, 0.0)).rgb * weight[i];
        result += texture(blurSampler, TexCoord - vec2(texOffset.x * i, 0.0)).rgb * weight[i];
    }

    return vec4(result, 1.0);
}

void main() {
    outColor = vec4(0.0);
    outBlur = blur(blurSampler, fragTexCoord);
}
