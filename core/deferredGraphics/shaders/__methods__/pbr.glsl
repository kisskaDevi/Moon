#ifndef PBR
#define PBR

#include "colorFunctions.glsl"

float geometricOcclusion(float NdotL, float NdotV, float k) {
    float attenuationL = NdotL / (k + (1.0 - k) * (NdotL));
    float attenuationV = NdotV / (k + (1.0 - k) * (NdotV));
    return attenuationL * attenuationV;
}
float microfacetDistribution(float NdotH, float alphaRoughness) {
    float roughnessSq = alphaRoughness * alphaRoughness;
    float f = (NdotH * roughnessSq - NdotH) * NdotH + 1.0;
    return roughnessSq / (pi * f * f);
}

vec3 diffuse(const in vec4 BaseColor, const in float metallic, const in vec3 f0) {
    vec3 diffuseColor = BaseColor.rgb * (vec3(1.0) - f0);
    diffuseColor *= 1.0 - metallic;

    return diffuseColor / pi;
}

vec3 specularReflection(vec3 specular, float DdotN) {
    return specular + (vec3(1.0) - specular) * pow(1.0 - DdotN, 5);
}

vec4 pbr(
    vec4 position,
    vec4 normal,
    vec4 baseColorTexture,
    vec4 eyePosition,
    vec4 lightColor,
    vec3 lightPosition
) {
    vec3 Direction      = normalize(eyePosition.xyz - position.xyz);
    vec3 LightDirection = normalize(lightPosition - position.xyz);
    vec3 Normal         = normal.xyz;
    vec3 H              = normalize(Direction + LightDirection);
    vec4 BaseColor      = SRGBtoLINEAR(baseColorTexture);

    float perceptualRoughness = decodeParameter(0x000000ff, 0, position.a) / 255.0f;
    float metallic = decodeParameter(0x0000ff00, 8, position.a) / 255.0f;

    vec3 f0 = vec3(0.04);

    vec3 specularColor = mix(f0, BaseColor.rgb, metallic);
    vec3 F = 0.1 * specularReflection(specularColor, clamp(dot(Direction, Normal), 0.0, 1.0));

    float alphaRoughness = (perceptualRoughness + 1) * (perceptualRoughness + 1) / 8.0;
    float G = geometricOcclusion(clamp(dot(Normal, LightDirection), 0.001, 1.0), clamp(abs(dot(Normal, Direction)), 0.001, 1.0), alphaRoughness);
    float D = microfacetDistribution(clamp(dot(Normal, H), 0.0, 1.0), perceptualRoughness);

    vec3 diffuseContrib = (1.0 - F) * diffuse(BaseColor, metallic, f0);
    vec3 specContrib = F * G * D / (4.0 * clamp(dot(Normal, LightDirection), 0.001, 1.0) * clamp(abs(dot(Normal, Direction)), 0.001, 1.0));

    vec4 outColor = vec4(clamp(dot(Normal, LightDirection), 0.001, 1.0) * lightColor.xyz * (diffuseContrib + specContrib), BaseColor.a);

    return outColor;
}

#endif
