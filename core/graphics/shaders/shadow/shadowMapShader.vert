#version 450
#define MAX_NUM_JOINTS 128

layout(set = 0, binding = 1) uniform LightUniformBufferObject
{
    mat4 projView;
    vec4 position;
    vec4 lightColor;
} lightubo;

layout (set = 1, binding = 0) uniform LocalUniformBuffer
{
    mat4 matrix;
} local;

layout (set = 2, binding = 0) uniform UBONode
{
    mat4 matrix;
    mat4 jointMatrix[MAX_NUM_JOINTS];
    float jointCount;
} node;

layout(location = 0)	in  vec3 inPosition;
layout(location = 1)	in  vec3 inNormal;
layout(location = 2)	in  vec2 inUV0;
layout(location = 3)	in  vec2 inUV1;
layout(location = 4)	in  vec4 inJoint0;
layout(location = 5)	in  vec4 inWeight0;
layout(location = 6)	in  vec3 inTangent;
layout(location = 7)	in  vec3 inBitangent;

void main()
{
    vec3 outPosition;
    if (node.jointCount > 0.0)
    {
	    // Mesh is skinned
	    mat4 skinMat =
	            inWeight0.x * node.jointMatrix[int(inJoint0.x)] +
	            inWeight0.y * node.jointMatrix[int(inJoint0.y)] +
	            inWeight0.z * node.jointMatrix[int(inJoint0.z)] +
	            inWeight0.w * node.jointMatrix[int(inJoint0.w)];

	    outPosition = vec3(local.matrix*node.matrix * skinMat* vec4(inPosition,1.0));
    } else
    {
	    outPosition = vec3(local.matrix*node.matrix * vec4(inPosition,1.0));
    }

    gl_Position = lightubo.projView * vec4(outPosition,1.0f);
}
