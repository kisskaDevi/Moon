#version 450

layout(set = 3, binding = 0) uniform sampler2D baseColorTexture;
layout(set = 3, binding = 1) uniform sampler2D metallicRoughnessTexture;
layout(set = 3, binding = 2) uniform sampler2D normalTexture;
layout(set = 3, binding = 3) uniform sampler2D occlusionTexture;
layout(set = 3, binding = 4) uniform sampler2D emissiveTexture;

layout(location = 0)	in vec4 position;
layout(location = 1)	in vec3 normal;
layout(location = 2)	in vec2 UV0;
layout(location = 3)	in vec2 UV1;
layout(location = 4)	in vec4 eyePosition;
layout(location = 5)	in mat3 TBN;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outBaseColor;
layout(location = 3) out vec4 outMetallicRoughness;
layout(location = 4) out vec4 outOcclusion;
layout(location = 5) out vec4 outEmissiveTexture;

layout (push_constant) uniform Material
{
    int normalTextureSet;
    int number;
} material;

vec3 getNormal()
{
    vec3 tangentNormal = normalize(texture(normalTexture, UV0).xyz * 2.0f - 1.0f);

    return normalize(TBN * tangentNormal);
}

void main()
{
    outPosition = position;
    outBaseColor = texture(baseColorTexture, UV0);
    outMetallicRoughness = texture(metallicRoughnessTexture, UV0);
    outNormal = vec4(material.normalTextureSet > -1 ? getNormal() : normalize(normal),material.number);
    outOcclusion = texture(occlusionTexture, UV0);
    outEmissiveTexture = texture(emissiveTexture, UV0);

    outPosition.a = 0.0f;
}
