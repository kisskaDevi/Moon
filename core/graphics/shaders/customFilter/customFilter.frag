#version 450

layout(set = 0, binding = 0) uniform sampler2D Sampler;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout (push_constant) uniform PC
{
    float dx;
    float dy;
} pc;

void main()
{
    outColor = vec4(0.0f);

    vec2 textel = 1.0f / textureSize(Sampler, 0);

    outColor += texture(Sampler,fragTexCoord + 1.5f*vec2(-textel.x*pc.dx,-textel.y*pc.dy));
    outColor += texture(Sampler,fragTexCoord + 1.5f*vec2(-textel.x*pc.dx, textel.y*pc.dy));
    outColor += texture(Sampler,fragTexCoord + 1.5f*vec2( textel.x*pc.dx,-textel.y*pc.dy));
    outColor += texture(Sampler,fragTexCoord + 1.5f*vec2( textel.x*pc.dx, textel.y*pc.dy));

    outColor += texture(Sampler,fragTexCoord + 3.0f*vec2(-textel.x*pc.dx,-textel.y*pc.dy));
    outColor += texture(Sampler,fragTexCoord + 3.0f*vec2(-textel.x*pc.dx, textel.y*pc.dy));
    outColor += texture(Sampler,fragTexCoord + 3.0f*vec2( textel.x*pc.dx,-textel.y*pc.dy));
    outColor += texture(Sampler,fragTexCoord + 3.0f*vec2( textel.x*pc.dx, textel.y*pc.dy));

    outColor /= 8.0f;
}
