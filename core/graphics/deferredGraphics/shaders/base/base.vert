#version 450
#define MAX_NUM_JOINTS 130

layout(set = 0, binding = 0) uniform GlobalUniformBuffer
{
    mat4 view;
    mat4 proj;
    vec4 eyePosition;
    float transparencyPass;
} global;

layout (set = 1, binding = 0) uniform LocalUniformBuffer
{
    mat4 matrix;
    vec4 constColor;
    vec4 colorFactor;
    vec4 bloomColor;
    vec4 bloomFactor;
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
layout(location = 4)	out vec3 outTangent;
layout(location = 5)	out vec3 outBitangent;
layout(location = 6)	out vec4 outConstColor;
layout(location = 7)	out vec4 outColorFactor;
layout(location = 8)	out vec4 outBloomColor;
layout(location = 9)	out vec4 outBloomFactor;
layout(location = 10)	out vec4 outEyePosition;
layout(location = 11)	out float depth;
layout(location = 12)	out vec4 glPosition;
layout(location = 13)	out float transparencyPass;

void main()
{
    outUV0 = inUV0;
    outUV1 = inUV1;
    outConstColor = local.constColor;
    outColorFactor = local.colorFactor;
    outBloomColor = local.bloomColor;
    outBloomFactor = local.bloomFactor;
    outEyePosition = global.eyePosition;
    transparencyPass = global.transparencyPass;

    mat4x4 model = local.matrix*node.matrix;
    if (node.jointCount > 0.0)
    {
	    // Mesh is skinned
	    mat4 skinMat =
	            inWeight0.x * node.jointMatrix[int(inJoint0.x)] +
	            inWeight0.y * node.jointMatrix[int(inJoint0.y)] +
	            inWeight0.z * node.jointMatrix[int(inJoint0.z)] +
	            inWeight0.w * node.jointMatrix[int(inJoint0.w)];

	    model *= skinMat;
	    outPosition	    =		     model * vec4(inPosition,	1.0);
	    outNormal	    = normalize(vec3(inverse(transpose(model)) * vec4(inNormal,	0.0)));
	    outTangent	    = normalize(vec3(model * vec4(inTangent,	0.0)));
	    outBitangent    = normalize(vec3(model * vec4(inBitangent,	0.0)));
    } else
    {
	    outPosition	    =		     model * vec4(inPosition,	1.0);
	    outNormal	    = normalize(vec3(inverse(transpose(model)) * vec4(inNormal,	0.0)));
	    outTangent	    = normalize(vec3(model * vec4(inTangent,	0.0)));
	    outBitangent    = normalize(vec3(model * vec4(inBitangent,	0.0)));
    }

    gl_Position = global.proj * global.view * outPosition;
    glPosition = gl_Position;
    glPosition /= glPosition.w;

    depth = gl_Position.z;
}
