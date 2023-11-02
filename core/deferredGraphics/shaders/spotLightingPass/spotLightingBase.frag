#include "../__methods__/lightDrop.glsl"
#include "../__methods__/pbr.glsl"
#include "../__methods__/shadow.glsl"
#include "../__methods__/geometricFunctions.glsl"

#ifndef SPOT_LIGHTING_BASE
#define SPOT_LIGHTING_BASE

layout(set = 1, binding = 0) uniform LightBufferObject {
    mat4 proj;
    mat4 view;
    mat4 projView;
    vec4 position;
    vec4 color;
    vec4 prop;
} light;

vec4 calcLight(
    const in vec4 position,
    const in vec4 normal,
    const in vec4 baseColorTexture,
    const in vec4 eyePosition,
    sampler2D shadowMap,
    sampler2D lightTexture
) {
    vec4 outColor = baseColorTexture;

    if(!checkZeroNormal(normal.xyz)) {        
        float type = light.prop.x;
        float lightPowerFactor = light.prop.y;
        float lightDropFactor = light.prop.z;

        vec4 lightProjView = light.projView * vec4(position.xyz, 1.0);
        float ShadowFactor = outsideSpotCondition(light.proj, light.view, type, position.xyz) ? 0.0 : shadowFactor(light.proj, shadowMap, lightProjView);

        vec4 lightTextureColor = texture(lightTexture, (lightProjView.xy / lightProjView.w) * 0.5 + 0.5);
        vec4 sumLightColor = vec4(max(light.color, lightTextureColor).xyz, 1.0);

        vec4 baseColor = pbr(position, normal, baseColorTexture, eyePosition, sumLightColor, light.position.xyz);
        float distribusion = (type == 0.0) ? lightDistribusion(position.xyz, light.position.xyz, light.proj, getDirection(light.view)) : 1.0;
        float lightDrop = lightDropFactor * lightDrop(length(light.position.xyz - position.xyz));
        float lightPower = lightPowerFactor * distribusion / (lightDrop > 0.0 ? lightDrop : 1.0);
        outColor = vec4(ShadowFactor * lightPower * baseColor.xyz, baseColor.a);
    }

    return outColor;
}

#endif