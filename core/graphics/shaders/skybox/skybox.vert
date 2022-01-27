#version 450
#define MAX_NUM_JOINTS 128
#define MAX_LIGHT_SOURCES 6

layout (set = 0, binding = 0) uniform UniformBuffer
{
    mat4 proj;
    mat4 view;
    mat4 model;
} local;

layout(location = 0)	in  vec3 inPosition;
layout(location = 1)	in  vec3 inNormal;
layout(location = 2)	in  vec2 inUV0;
layout(location = 3)	in  vec2 inUV1;
layout(location = 4)	in  vec4 inJoint0;
layout(location = 5)	in  vec4 inWeight0;
layout(location = 6)	in  vec3 inTangent;
layout(location = 7)	in  vec3 inBitangent;

layout(location = 0)	out vec3 outUVW;

void main()
{
    outUVW = inPosition;
    gl_Position = local.proj * local.view * local.model * vec4(inPosition.xyz,1.0f);
}
