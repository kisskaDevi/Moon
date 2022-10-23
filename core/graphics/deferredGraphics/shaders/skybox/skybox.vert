#version 450

layout (set = 0, binding = 0) uniform UniformBuffer
{
    mat4 proj;
    mat4 view;
    mat4 model;
} local;

layout(location = 0)	out vec3 outUVW;
layout(location = 1)	out float depth;

vec3 vertex[36] = vec3[](
    vec3(-1.0f,-1.0f,-1.0f),
    vec3(-1.0f,-1.0f, 1.0f),
    vec3(-1.0f, 1.0f, 1.0f),
    vec3( 1.0f, 1.0f,-1.0f),
    vec3(-1.0f,-1.0f,-1.0f),
    vec3(-1.0f, 1.0f,-1.0f),
    vec3( 1.0f,-1.0f, 1.0f),
    vec3(-1.0f,-1.0f,-1.0f),
    vec3( 1.0f,-1.0f,-1.0f),
    vec3( 1.0f, 1.0f,-1.0f),
    vec3( 1.0f,-1.0f,-1.0f),
    vec3(-1.0f,-1.0f,-1.0f),
    vec3(-1.0f,-1.0f,-1.0f),
    vec3(-1.0f, 1.0f, 1.0f),
    vec3(-1.0f, 1.0f,-1.0f),
    vec3( 1.0f,-1.0f, 1.0f),
    vec3(-1.0f,-1.0f, 1.0f),
    vec3(-1.0f,-1.0f,-1.0f),
    vec3(-1.0f, 1.0f, 1.0f),
    vec3(-1.0f,-1.0f, 1.0f),
    vec3( 1.0f,-1.0f, 1.0f),
    vec3( 1.0f, 1.0f, 1.0f),
    vec3( 1.0f,-1.0f,-1.0f),
    vec3( 1.0f, 1.0f,-1.0f),
    vec3( 1.0f,-1.0f,-1.0f),
    vec3( 1.0f, 1.0f, 1.0f),
    vec3( 1.0f,-1.0f, 1.0f),
    vec3( 1.0f, 1.0f, 1.0f),
    vec3( 1.0f, 1.0f,-1.0f),
    vec3(-1.0f, 1.0f,-1.0f),
    vec3( 1.0f, 1.0f, 1.0f),
    vec3(-1.0f, 1.0f,-1.0f),
    vec3(-1.0f, 1.0f, 1.0f),
    vec3( 1.0f, 1.0f, 1.0f),
    vec3(-1.0f, 1.0f, 1.0f),
    vec3( 1.0f,-1.0f, 1.0f)
);

void main()
{
    vec3 Position = vertex[gl_VertexIndex];
    outUVW = Position;
    gl_Position = local.proj * local.view * local.model * vec4(Position,1.0f);

    depth = gl_Position.z;
}
