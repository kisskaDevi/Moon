#version 450
#define transparentLayersCount 3

const float pi = 3.141592653589793f;

layout(set = 0, binding = 0) uniform GlobalUniformBuffer
{
    mat4 view;
    mat4 proj;
    vec4 eyePosition;
} global;
layout(set = 0, binding = 1) uniform sampler2D Sampler;
layout(set = 0, binding = 2) uniform sampler2D bloomSampler;
layout(set = 0, binding = 3) uniform sampler2D position;
layout(set = 0, binding = 4) uniform sampler2D normal;
layout(set = 0, binding = 5) uniform sampler2D depth;
layout(set = 0, binding = 6) uniform sampler2D layersSampler[transparentLayersCount];
layout(set = 0, binding = 7) uniform sampler2D layersBloomSampler[transparentLayersCount];
layout(set = 0, binding = 8) uniform sampler2D layersPosition[transparentLayersCount];
layout(set = 0, binding = 9) uniform sampler2D layersNormal[transparentLayersCount];
layout(set = 0, binding = 10) uniform sampler2D layersDepth[transparentLayersCount];

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outBloom;

mat4 projview = global.proj * global.view;

float h = 0.3f;
vec3 n = vec3(1.33f, 1.33f + 1.0f, 1.33f + 2.0f);

vec3 findRefrCoords(const in vec3 layerPointPosition, const in vec3 layerPointNormal, float n, vec3 startPos){
    vec3 beamDirection = normalize(layerPointPosition - startPos);
    float cosAlpha = - dot(layerPointNormal,beamDirection);
    float sinAlpha = sqrt(1.0f - cosAlpha * cosAlpha);

    float deviation = h * sinAlpha * (1.0f - cosAlpha / sqrt(n*n - sinAlpha*sinAlpha));
    vec3 direction = layerPointNormal + beamDirection * cosAlpha;
    direction = - normalize(direction);

    vec4 dpos = projview * vec4(layerPointPosition + deviation * direction,1.0f);

    return vec3(dpos.xy/dpos.w * 0.5f + 0.5f,dpos.z/dpos.w);
}
vec3 layerPointPosition(const in int i, const in vec2 coord){
    return texture(layersPosition[i], coord).xyz;
}
vec3 layerPointNormal(const in int i, const in vec2 coord){
    return normalize(texture(layersNormal[i],coord).xyz);
}
vec4 layerColor(const in int i, const in vec2 coord){
    return texture(layersSampler[i],coord);
}
bool insideCond(const in vec2 coords){
    return coords.x<1.0f&&coords.y<1.0f&&coords.x>0.0f&&coords.y>0.0f;
}
bool depthCond(float z, vec2 coords){
    return z < texture(depth,coords.xy).r;
}

vec4 findRefrColor(const int i, inout vec3 coords, const float n, vec3 startPos)
{
    vec4 color = vec4(0.0f);
    if(insideCond(coords.xy) && depthCond(texture(layersDepth[i],coords.xy).r,coords.xy))
    {
        color = layerColor(i,coords.xy);
        coords  = findRefrCoords(layerPointPosition(i,coords.xy), layerPointNormal(i,coords.xy),n,startPos);
    }
    return color;
}

void main()
{
    vec3 coordsR = vec3(fragTexCoord,0.0f);
    vec3 coordsG = vec3(fragTexCoord,0.0f);
    vec3 coordsB = vec3(fragTexCoord,0.0f);

    vec4 refrColor = vec4(0.0f);
    vec4 refrBloom = vec4(0.0f);
    vec3 startPosR = global.eyePosition.xyz;
    vec3 startPosG = global.eyePosition.xyz;
    vec3 startPosB = global.eyePosition.xyz;
    vec2 oldCoordsR = fragTexCoord;
    vec2 oldCoordsG = fragTexCoord;
    vec2 oldCoordsB = fragTexCoord;
    for(int i=0;i<transparentLayersCount;i++)
    {
        vec4 newBloom = vec4(0.0f);
        if(insideCond(coordsR.xy) && depthCond(texture(layersDepth[i],coordsR.xy).r,coordsR.xy))
            newBloom.r = texture(layersBloomSampler[i],coordsR.xy).r;
        if(insideCond(coordsG.xy) && depthCond(texture(layersDepth[i],coordsG.xy).r,coordsG.xy))
            newBloom.g = texture(layersBloomSampler[i],coordsG.xy).g;
        if(insideCond(coordsB.xy) && depthCond(texture(layersDepth[i],coordsB.xy).r,coordsB.xy))
            newBloom.b = texture(layersBloomSampler[i],coordsB.xy).b;
        refrBloom = max(refrBloom,newBloom);

        vec4 newColor = vec4(findRefrColor(i,coordsR,n.r,startPosR).r, findRefrColor(i,coordsG,n.g,startPosG).g, findRefrColor(i,coordsB,n.b,startPosB).b, 0.0f);
        refrColor = max(refrColor,newColor);
        startPosR = layerPointPosition(i,oldCoordsR.xy);
        startPosG = layerPointPosition(i,oldCoordsG.xy);
        startPosB = layerPointPosition(i,oldCoordsB.xy);
        oldCoordsR = coordsR.xy;
        oldCoordsG = coordsG.xy;
        oldCoordsB = coordsB.xy;
    }
    outColor = vec4(    insideCond(coordsR.xy) && depthCond(coordsR.z,coordsR.xy) ? texture(Sampler,coordsR.xy).r : 0.0f,
                        insideCond(coordsG.xy) && depthCond(coordsG.z,coordsG.xy) ? texture(Sampler,coordsG.xy).g : 0.0f,
                        insideCond(coordsB.xy) && depthCond(coordsB.z,coordsB.xy) ? texture(Sampler,coordsB.xy).b : 0.0f, 0.0f);
    outColor = max(refrColor,outColor);

    outBloom = vec4(    insideCond(coordsR.xy) && depthCond(coordsR.z,coordsR.xy) ? texture(bloomSampler,coordsR.xy).r : 0.0f,
                        insideCond(coordsG.xy) && depthCond(coordsG.z,coordsG.xy) ? texture(bloomSampler,coordsG.xy).g : 0.0f,
                        insideCond(coordsB.xy) && depthCond(coordsB.z,coordsB.xy) ? texture(bloomSampler,coordsB.xy).b : 0.0f, 0.0f);
    outBloom = max(refrBloom,outBloom);
}
