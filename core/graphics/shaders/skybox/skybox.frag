#version 450

layout(set = 0, binding = 1)	uniform samplerCube samplerCubeMap;

layout(location = 0)	in vec3 inUVW;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outBaseColor;
layout(location = 3) out vec4 outMetallicRoughness;
layout(location = 4) out vec4 outOcclusion;
layout(location = 5) out vec4 outEmissiveTexture;

//===================================================main====================================================================//

void main()
{
    outPosition = vec4(inUVW,1.0f);
    outBaseColor = texture(samplerCubeMap, inUVW);
    outMetallicRoughness = vec4(0.0f);
    outNormal = vec4(0.0f);
    outOcclusion = vec4(0.0f);
    outEmissiveTexture = vec4(0.0f);

    outPosition.a = 2.0f;
}
