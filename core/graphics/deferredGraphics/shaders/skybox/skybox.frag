#version 450

layout(set = 0, binding = 1)	uniform samplerCube samplerCubeMap;

layout(location = 0)	in vec3 inUVW;
layout(location = 1)	in float depth;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outBaseColor;
layout(location = 3) out vec4 outEmissiveTexture;

//===================================================main====================================================================//

void main()
{
    outPosition = vec4(inUVW,depth);
    outBaseColor = texture(samplerCubeMap, inUVW);
    outNormal = vec4(0.0f);
    outEmissiveTexture = vec4(0.0f);
}
