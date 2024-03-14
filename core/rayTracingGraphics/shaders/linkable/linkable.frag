#version 450

layout(set = 0, binding = 0) uniform sampler2D Sampler;
layout(set = 0, binding = 1) uniform sampler2D bbSampler;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(Sampler, fragTexCoord);
    outColor += texture(bbSampler, fragTexCoord);
}
