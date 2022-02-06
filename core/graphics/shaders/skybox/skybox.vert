#version 450
#define MAX_LIGHT_SOURCES 6

layout (set = 0, binding = 0) uniform UniformBuffer
{
    mat4 proj;
    mat4 view;
    mat4 model;
} local;

layout(location = 0)	in  vec3 inPosition;

layout(location = 0)	out vec3 outUVW;

void main()
{
    outUVW = inPosition;
    gl_Position = local.proj * local.view * local.model * vec4(inPosition.xyz,1.0f);
}
