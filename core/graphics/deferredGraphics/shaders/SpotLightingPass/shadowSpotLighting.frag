#version 450
#define MANUAL_SRGB 1
#define pi 3.141592653589793f

layout (push_constant) uniform PC
{
    float minAmbientFactor;
}pc;

layout(set = 1, binding = 0) uniform LightBufferObject
{
    mat4 proj;
    mat4 view;
    mat4 projView;
    vec4 position;
    vec4 lightColor;
    vec4 lightProp;
}light;

layout(location = 0)	in vec4 eyePosition;
layout(location = 1)	in vec2 fragTexCoord;
layout(location = 2)	in vec4 glPosition;

layout(location = 3)	flat in vec3 lightPosition;
layout(location = 4)	flat in vec4 lightColor;
layout(location = 5)	flat in vec4 lightProp;
layout(location = 6)	flat in mat4 lightProjView;
layout(location = 10)	flat in mat4 projview;

layout(input_attachment_index = 0, binding = 0) uniform subpassInput inPositionTexture;
layout(input_attachment_index = 1, binding = 1) uniform subpassInput inNormalTexture;
layout(input_attachment_index = 2, binding = 2) uniform subpassInput inBaseColorTexture;
layout(input_attachment_index = 3, binding = 3) uniform subpassInput inEmissiveTexture;
layout(input_attachment_index = 4, binding = 4) uniform subpassInput inDepthTexture;

layout(set = 1, binding = 1) uniform sampler2D shadowMap;
layout(set = 1, binding = 2) uniform sampler2D lightTexture;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outBlur;
layout(location = 2) out vec4 outBloom;

#include "metods/outsideSpotCondition.frag"
#include "metods/lightDrop.frag"
#include "metods/pbr.frag"
#include "metods/shadow.frag"

void main()
{
    vec4 position = subpassLoad(inPositionTexture);
    vec4 normal = subpassLoad(inNormalTexture);
    vec4 baseColorTexture = subpassLoad(inBaseColorTexture);
    vec4 emissiveTexture = subpassLoad(inEmissiveTexture);
    float depthMap = subpassLoad(inDepthTexture).r;

    outColor = vec4(0.0f,0.0f,0.0f,0.0f);
    outBlur = vec4(0.0f,0.0f,0.0f,0.0f);
    outBloom = vec4(0.0f,0.0f,0.0f,0.0f);

    float type = lightProp.x;
    float lightPowerFactor = lightProp.y;
    float lightDropFactor = lightProp.z;

    vec4 fragLightPosition = lightProjView * vec4(position.xyz,1.0f);
    float ShadowFactor = outsideSpotCondition(fragLightPosition.xyz/ fragLightPosition.w,type) ? 0.0f : shadowFactor(shadowMap, fragLightPosition);

    vec3 lightDirection =  - normalize(vec3(light.view[0][2],light.view[1][2],light.view[2][2]));
    float distribusion = (type == 0.0) ? lightDistribusion(position.xyz,lightPosition.xyz,light.proj,lightDirection) : 1.0f;
    vec4 textureLightColor = distribusion * texture(lightTexture, (fragLightPosition.xy / fragLightPosition.w) * 0.5f + 0.5f);
    vec4 sumLightColor = vec4(max(lightColor.x,textureLightColor.x),max(lightColor.y,textureLightColor.y),max(lightColor.z,textureLightColor.z),1.0f);

    vec4 baseColor = SRGBtoLINEAR(baseColorTexture);
    if(!(normal.x==0.0f&&normal.y==0.0f&&normal.z==0.0f)){
        baseColor = PBR(position,normal,baseColorTexture,eyePosition,sumLightColor,lightPosition);
        float lightDrop = lightDropFactor * lightDrop(length(lightPosition - position.xyz));
        float lightPower = lightPowerFactor / lightDrop;
        outColor += vec4(ShadowFactor * lightPower * baseColor.xyz, baseColor.a);
    }

    if(outColor.x>0.95f&&outColor.y>0.95f&&outColor.z>0.95f)    outBloom += outColor;
    else                                                        outBloom += vec4(0.0f,0.0f,0.0f,1.0f);

    //color = mix(color, color * emissiveTexture.a, 1.0f);  ???

    outColor += SRGBtoLINEAR(emissiveTexture);
    outBloom += SRGBtoLINEAR(emissiveTexture);
}
