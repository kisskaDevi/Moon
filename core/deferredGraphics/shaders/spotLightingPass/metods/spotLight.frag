#include "outsideSpotCondition.frag"
#include "lightDrop.frag"
#include "pbr.frag"
#include "shadow.frag"

layout(set = 1, binding = 0) uniform LightBufferObject
{
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
    sampler2D lightTexture,
    bool enableShadow)
{
    vec4 outColor = vec4(0.0f,0.0f,0.0f,0.0f);

    float type = light.prop.x;
    float lightPowerFactor = light.prop.y;
    float lightDropFactor = light.prop.z;

    vec4 lightView = light.view * vec4(position.xyz,1.0f);
    lightView *= vec4(- light.proj[1][1] / light.proj[0][0], 1.0f, -1.0f, 1.0f);
    vec4 lightProjView = light.projView * vec4(position.xyz,1.0f);
    float ShadowFactor = outsideSpotCondition(lightView.xyz,type) ? 0.0f : (enableShadow ? shadowFactor(shadowMap, lightProjView) : 1.0f);

    vec3 lightDirection = - normalize(vec3(light.view[0][2],light.view[1][2],light.view[2][2]));
    float distribusion = (type == 0.0) ? lightDistribusion(position.xyz,light.position.xyz,light.proj,lightDirection) : 1.0f;
    vec4 lightTextureColor = distribusion * texture(lightTexture, (lightProjView.xy / lightProjView.w) * 0.5f + 0.5f);
    vec4 sumLightColor = vec4(max(light.color.x,lightTextureColor.x),max(light.color.y,lightTextureColor.y),max(light.color.z,lightTextureColor.z),1.0f);

    vec4 baseColor = SRGBtoLINEAR(baseColorTexture);
    if(!(normal.x==0.0f&&normal.y==0.0f&&normal.z==0.0f)){
        baseColor = PBR(position,normal,baseColorTexture,eyePosition,sumLightColor,light.position.xyz);
        float lightDrop = lightDropFactor * lightDrop(length(light.position.xyz - position.xyz));
        float lightPower = lightPowerFactor / (lightDrop > 0.0f ? lightDrop : 1.0f);
        outColor += vec4(ShadowFactor * lightPower * baseColor.xyz, baseColor.a);
    }

    return outColor;
}

bool checkBrightness(vec4 color)
{
    return color.x > 0.95f && color.y > 0.95f && color.z > 0.95f;
}