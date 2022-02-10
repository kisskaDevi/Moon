#version 450
#define MAX_NUM_JOINTS 130

layout(set = 0, binding = 0) uniform GlobalUniformBuffer
{
    mat4 view;
    mat4 proj;
    vec4 eyePosition;
} global;

layout (set = 1, binding = 0) uniform LocalUniformBuffer
{
    mat4 matrix;
    vec4 color;
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

layout(location = 0)	out vec4 outPosition;
layout(location = 1)	out vec3 outNormal;
layout(location = 2)	out vec2 outUV0;
layout(location = 3)	out vec2 outUV1;
layout(location = 4)	out vec4 eyePosition;
layout(location = 5)	out mat3 TBN;

void main()
{
    outUV0 = inUV0;
    outUV1 = inUV1;
    eyePosition = global.eyePosition;

    mat4x4 model = local.matrix*node.matrix;
    vec3 tangent, bitangent;
    if (node.jointCount > 0.0)
    {
	    // Mesh is skinned
	    mat4 skinMat =
	            inWeight0.x * node.jointMatrix[int(inJoint0.x)] +
	            inWeight0.y * node.jointMatrix[int(inJoint0.y)] +
	            inWeight0.z * node.jointMatrix[int(inJoint0.z)] +
	            inWeight0.w * node.jointMatrix[int(inJoint0.w)];

	    model *= skinMat;
	    outPosition	    =		model * vec4(inPosition,1.0);
	    mat3 mNormal = transpose(inverse(mat3(model)));
	    outNormal	    = normalize(vec3(mNormal * inNormal));
	    tangent	    = normalize(vec3(mNormal * inTangent));
	    bitangent	    = normalize(vec3(mNormal * inBitangent));
    } else
    {
	    outPosition	    =		model * vec4(inPosition,1.0);
	    mat3 mNormal = transpose(inverse(mat3(model)));
	    outNormal	    = normalize(vec3(mNormal * inNormal));
	    tangent	    = normalize(vec3(mNormal * inTangent));
	    bitangent	    = normalize(vec3(mNormal * inBitangent));
    }

    TBN = mat3(tangent, bitangent, outNormal);

    gl_Position = global.proj * global.view * outPosition;
}
