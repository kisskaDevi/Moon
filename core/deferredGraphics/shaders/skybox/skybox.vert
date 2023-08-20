#version 450

#include "../__methods__/defines.glsl"

layout (set = 0, binding = 0) uniform UniformBuffer
{
    mat4 view;
    mat4 proj;
    vec4 eyePosition;
} camera;

layout (set = 1, binding = 0) uniform LocalUniformBuffer
{
    mat4 matrix;
    vec4 constColor;
    vec4 colorFactor;
    vec4 bloomColor;
    vec4 bloomFactor;
} local;

layout(location = 0)	out vec3 outUVW;
layout(location = 1)	out float depth;
layout(location = 2)	out vec4 constColor;
layout(location = 3)	out vec4 colorFactor;

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
    constColor = local.constColor;
    colorFactor = local.colorFactor;

    mat4x4 cameraModel = mat4x4(1.0f);
    cameraModel[3][0] = camera.eyePosition.x;
    cameraModel[3][1] = camera.eyePosition.y;
    cameraModel[3][2] = camera.eyePosition.z;

    vec3 Position = vertex[gl_VertexIndex];
    outUVW = vec4(vec4(Position,1.0f)).xyz;
    gl_Position = camera.proj * camera.view * cameraModel * local.matrix * vec4(Position,1.0f);

    depth = gl_Position.z;
}
