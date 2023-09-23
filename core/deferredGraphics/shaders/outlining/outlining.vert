#version 450

#include "../__methods__/defines.glsl"
#define MAX_NUM_JOINTS 128

layout(set = 0, binding = 0) uniform GlobalUniformBuffer {
    mat4 view;
    mat4 proj;
    vec4 eyePosition;
} global;

layout(set = 1, binding = 0) uniform LocalUniformBuffer {
    mat4 matrix;
} local;

layout(set = 2, binding = 0) uniform UBONode {
    mat4 matrix;
    mat4 jointMatrix[MAX_NUM_JOINTS];
    float jointCount;
} node;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV0;
layout(location = 3) in vec2 inUV1;
layout(location = 4) in vec4 inJoint0;
layout(location = 5) in vec4 inWeight0;
layout(location = 6) in vec3 inTangent;
layout(location = 7) in vec3 inBitangent;

layout(push_constant) uniform Stencil {
    vec4 color;
    float width;
} stencil;

layout(location = 0) out vec4 outPosition;

void main() {
    outPosition = vec4(inPosition.xyz, 1.0);

    mat4 skinMat =  node.jointCount > 0.0 ?  
                    inWeight0.x * node.jointMatrix[int(inJoint0.x)] +
                    inWeight0.y * node.jointMatrix[int(inJoint0.y)] +
                    inWeight0.z * node.jointMatrix[int(inJoint0.z)] +
                    inWeight0.w * node.jointMatrix[int(inJoint0.w)] : mat4(1.0f);
    mat4x4 model = local.matrix * node.matrix * skinMat;

    outPosition = model * outPosition;
    vec3 Normal = normalize(vec3(inverse(transpose(model)) * vec4(inNormal, 0.0)));
    outPosition = vec4(outPosition.xyz + Normal * stencil.width, 1.0);

    gl_Position = global.proj * global.view * outPosition;
}
