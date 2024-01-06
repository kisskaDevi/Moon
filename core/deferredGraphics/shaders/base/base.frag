#version 450

#include "../__methods__/defines.glsl"
#include "../__methods__/colorFunctions.glsl"

const uint PBR_WORKFLOW_METALLIC_ROUGHNESS = 0;
const uint PBR_WORKFLOW_SPECULAR_GLOSINESS = 1;
const float c_MinRoughness = 0.04;

layout(constant_id = 0) const bool enableTransparencyPass = false;

layout(set = 0, binding = 1) uniform samplerCube samplerCubeMap;
layout(set = 0, binding = 2) uniform sampler2D depthMap;

layout (push_constant) uniform Material{
    vec4 baseColorFactor;
    vec4 emissiveFactor;
    vec4 diffuseFactor;
    vec4 specularFactor;
    float workflow;
    int baseColorTextureSet;
    int physicalDescriptorTextureSet;
    int normalTextureSet;
    int occlusionTextureSet;
    int emissiveTextureSet;
    float metallicFactor;
    float roughnessFactor;
    float alphaMask;
    float alphaMaskCutoff;
    int number;
} material;

layout(set = 3, binding = 0) uniform sampler2D baseColorTexture;
layout(set = 3, binding = 1) uniform sampler2D metallicRoughnessTexture;
layout(set = 3, binding = 2) uniform sampler2D normalTexture;
layout(set = 3, binding = 3) uniform sampler2D occlusionTexture;
layout(set = 3, binding = 4) uniform sampler2D emissiveTexture;

layout(location = 0)	in vec4 position;
layout(location = 1)	in vec3 normal;
layout(location = 2)	in vec2 UV0;
layout(location = 3)	in vec2 UV1;
layout(location = 4)	in vec3 tangent;
layout(location = 5)	in vec3 bitangent;
layout(location = 6)	in vec4 eyePosition;
layout(location = 7)	in vec4 glPosition;

layout (set = 1, binding = 0) uniform LocalUniformBuffer
{
    mat4 matrix;
    vec4 constColor;
    vec4 colorFactor;
    vec4 bloomColor;
    vec4 bloomFactor;
} local;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outBaseColor;

vec3 getNormal() {
    vec3 tangentNormal = normalize(texture(normalTexture, material.normalTextureSet == 0 ? UV0 : UV1).xyz * 2.0 - 1.0);
    mat3 TBN = mat3(tangent, bitangent, normal);
    return normalize(TBN * tangentNormal);
}

float convertMetallic(vec3 diffuse, vec3 specular, float maxSpecular) {
    float perceivedDiffuse = sqrt(0.299 * diffuse.r * diffuse.r + 0.587 * diffuse.g * diffuse.g + 0.114 * diffuse.b * diffuse.b);
    float perceivedSpecular = sqrt(0.299 * specular.r * specular.r + 0.587 * specular.g * specular.g + 0.114 * specular.b * specular.b);
    if(perceivedSpecular < c_MinRoughness) {
        return 0.0;
    }
    float a = c_MinRoughness;
    float b = perceivedDiffuse * (1.0 - maxSpecular) / (1.0 - c_MinRoughness) + perceivedSpecular - 2.0 * c_MinRoughness;
    float c = c_MinRoughness - perceivedSpecular;
    float D = max(b * b - 4.0 * a * c, 0.0);
    return clamp((-b + sqrt(D)) / (2.0 * a), 0.0, 1.0);
}

void metallicRoughnessWorkflow(inout float perceptualRoughness, inout float metallic, inout vec4 baseColor) {
    if (material.physicalDescriptorTextureSet > -1) {
        vec4 mrSample = texture(metallicRoughnessTexture, UV0); // r - (optional) occlusion map, g - roughness, b - metallic
        perceptualRoughness = mrSample.g * material.roughnessFactor;
        metallic = mrSample.b * material.metallicFactor;
    } else {
        perceptualRoughness = clamp(material.roughnessFactor, c_MinRoughness, 1.0);
        metallic = clamp(material.metallicFactor, 0.0, 1.0);
    }

    baseColor = material.baseColorFactor * ((material.baseColorTextureSet > -1) ? baseColor : vec4(1.0f));
}

void specularGlosinessWorkflow(inout float perceptualRoughness, inout float metallic, inout vec4 baseColor) {
    vec4 diffuse = baseColor;
    vec3 specular = SRGBtoLINEAR(texture(metallicRoughnessTexture, UV0)).rgb;
    float maxSpecular = max(max(specular.r, specular.g), specular.b);

    perceptualRoughness = (material.physicalDescriptorTextureSet > -1) ? (1.0 - texture(metallicRoughnessTexture,UV0).a) : 0.0;
    metallic = convertMetallic(diffuse.rgb, specular, maxSpecular);

    const float epsilon = 1e-6;

    vec3 baseColorDiffusePart = diffuse.rgb * ((1.0 - maxSpecular) / (1 - c_MinRoughness) / max(1 - metallic, epsilon)) * material.diffuseFactor.rgb;
    vec3 baseColorSpecularPart = specular - (vec3(c_MinRoughness) * (1 - metallic) * (1 / max(metallic, epsilon))) * material.specularFactor.rgb;
    baseColor = vec4(mix(baseColorDiffusePart, baseColorSpecularPart, metallic * metallic), diffuse.a);
}

void main()
{
    float perceptualRoughness;
    float metallic;
    vec4 baseColor = SRGBtoLINEAR(vec4(local.colorFactor.xyz, 1.0f) * texture(baseColorTexture, UV0) + local.constColor);

    if(!enableTransparencyPass && baseColor.a < 1.0f){
        discard;
    }
    if(enableTransparencyPass &&
        (baseColor.a >= 1.0f || baseColor.a <= 0.0f || glPosition.z / glPosition.w < 1.001f * texture(depthMap , glPosition.xy / glPosition.w * 0.5f + 0.5f).r)){
        discard;
    }

//    vec3 I = normalize(position.xyz - eyePosition.xyz);
//    vec3 R = reflect(I, outNormal.xyz);
//    vec4 reflection = texture(samplerCubeMap, R);
//    outBaseColor = vec4(max(outBaseColor.r,reflection.r),max(outBaseColor.g,reflection.g),max(outBaseColor.b,reflection.b), outBaseColor.a);

    switch(uint(material.workflow)) {
        case PBR_WORKFLOW_METALLIC_ROUGHNESS: {
            metallicRoughnessWorkflow(perceptualRoughness, metallic, baseColor);
            break;
        }
        case PBR_WORKFLOW_SPECULAR_GLOSINESS: {
            specularGlosinessWorkflow(perceptualRoughness, metallic, baseColor);
            break;
        }
    }

    uint PerceptualRoughness = uint(255.0f * perceptualRoughness);
    uint Metallic = uint(255.0f * metallic);
    uint number = uint(material.number);
    float params = uintBitsToFloat((PerceptualRoughness << 0) | (Metallic << 8) | (number << 16));
    
    float ao = material.occlusionTextureSet > -1 ? texture(occlusionTexture,UV0).r : 1.0f;
    vec4 bloomColor = vec4(local.bloomFactor.xyz, 1.0f) * texture(emissiveTexture,  UV0) + local.bloomColor;
    float emissiveAndAO = codeToFloat(bloomColor.xyz, ao);

    outBaseColor = baseColor;
    outPosition  = vec4(position.xyz, params);
    outNormal    = vec4(material.normalTextureSet > -1 ? getNormal() : normal, emissiveAndAO);
}
