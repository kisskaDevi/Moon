#version 450

layout(set = 0, binding = 8) uniform GlobalUniformBuffer
{
    vec4 eyePosition;
} global;

layout(location = 0)	out vec4 eyePosition;
layout(location = 1)	out vec2 fragTexCoord;

vec2 positions[6] = vec2[](
    vec2(-1.0f, -1.0f),
    vec2( 1.0f, -1.0f),
    vec2( 1.0f,  1.0f),
    vec2(1.0f, 1.0f),
    vec2(-1.0f, 1.0f),
    vec2( -1.0f,  -1.0f)
);

vec2 fragCoord[6] = vec2[](
    vec2(0.0f, 0.0f),
    vec2(1.0f, 0.0f),
    vec2(1.0f, 1.0f),
    vec2(1.0f, 1.0f),
    vec2(0.0f, 1.0f),
    vec2(0.0f, 0.0f)
);

void main()
{
    fragTexCoord = fragCoord[gl_VertexIndex];
    eyePosition = global.eyePosition;
    gl_Position = vec4(positions[gl_VertexIndex],0.0, 1.0);
}
