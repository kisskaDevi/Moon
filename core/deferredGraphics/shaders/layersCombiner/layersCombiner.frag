#version 450

const float pi = 3.141592653589793f;

layout (constant_id = 0) const int transparentLayersCount = 1;

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
layout(set = 0, binding = 11) uniform sampler2D skybox;
layout(set = 0, binding = 12) uniform sampler2D skyboxBloom;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outBloom;

mat4 projview = global.proj * global.view;
vec3 eyePosition = global.eyePosition.xyz;

float h = 0.6f;
float nbegin = 1.33f;
float nend = nbegin + 2.0f;
float reflectionProbability = 0.04f;

vec3 findRefrCoords(const in vec3 startPos, const in vec3 layerPointPosition, const in vec3 layerPointNormal, float n){
    vec3 beamDirection = normalize(layerPointPosition - startPos);
    float cosAlpha = - dot(layerPointNormal,beamDirection);
    float sinAlpha = sqrt(1.0f - cosAlpha * cosAlpha);

    float deviation = h * sinAlpha * (1.0f - cosAlpha / sqrt(n*n - sinAlpha*sinAlpha));
    vec3 direction = - normalize(layerPointNormal + beamDirection * cosAlpha);
    vec4 position = projview * vec4(layerPointPosition + deviation * direction, 1.0f);

    return vec3(position.xy/position.w * 0.5f + 0.5f, position.z/position.w);
}

vec3 layerPointPosition(const in int i, const in vec2 coord){
    return texture(layersPosition[i], coord).xyz;
}

vec3 layerPointNormal(const in int i, const in vec2 coord){
    return normalize(texture(layersNormal[i],coord).xyz);
}

float layerDepth(const in int i, const in vec2 coord){
    return texture(layersDepth[i],coord).r;
}

bool insideCond(const in vec2 coords){
    return (coords.x <= 1.0f) && (coords.y <= 1.0f) && (coords.x >= 0.0f) && (coords.y >= 0.0f);
}

bool depthCond(float z, vec2 coords){
    return z <= texture(depth,coords.xy).r;
}

void findRefr(const int i, const float n, inout vec3 startPos, inout vec3 coords)
{
    if(insideCond(coords.xy) && depthCond(layerDepth(i,coords.xy),coords.xy))
    {
        vec3 start = startPos;
        startPos = layerPointPosition(i,coords.xy);
        if(layerDepth(i,coords.xy)!=1.0f){
            coords  = findRefrCoords(start, layerPointPosition(i,coords.xy), layerPointNormal(i,coords.xy), n);
        }
    }
}

vec4 findColor(const in vec3 coord, sampler2D Sampler, sampler2D skybox, bool enableSkybox){
    vec4 skyboxColor = enableSkybox && texture(depth,coord.xy).r == 1.0f ? texture(skybox,coord.xy) : vec4(0.0f);
    return (insideCond(coord.xy) ? texture(Sampler,coord.xy) + skyboxColor : vec4(0.0f));
}

vec4 accumulateColor(vec3 beginCoords, vec3 endCoords, float step, sampler2D Sampler, sampler2D Depth, sampler2D skybox, bool enableSkybox){
    vec4 color = vec4(0.0f);
    for(float t = 0.0f; t < 1.0f; t += step){
        vec3 coords = beginCoords + (endCoords - beginCoords) * t;
        vec4 factor = vec4(4.0f * abs(t - 0.5) - 2.0f / 3.0f,
                           1.0f - abs(2.0f * t - 2.0f / 3.0f),
                           1.0f - abs(2.0f * t - 4.0f / 3.0f), 
                           1.0f);
        float layDepth = texture(Depth,coords.xy).r;
        if(depthCond(layDepth,coords.xy) && coords.z <= layDepth)
        {
            color += (beginCoords != vec3(fragTexCoord,0.0f) ? factor : vec4(1.0f)) * findColor(coords, Sampler, skybox, enableSkybox);
        }
    }
    return color;
}

void main()
{
    bool check = (texture(layersSampler[0],fragTexCoord.xy).a == 0.0f);

    float step = check ? 1.0f : 0.02f ;
    float incrementStep = 2.0f;

    vec3 beginCoords = vec3(fragTexCoord,0.0f), beginStartPos = eyePosition;
    vec3 endCoords = vec3(fragTexCoord,0.0f), endStartPos = eyePosition;
    vec4 layerColor = vec4(0.0f), layerBloom = vec4(0.0f);

    for(int i = 0; i < (check ? 0 : transparentLayersCount); i++)
    {
        float layerStep = step * (pow(incrementStep,i));
        vec4 color = (i==0 ? reflectionProbability : 1.0f) * accumulateColor(beginCoords,endCoords,layerStep,layersSampler[i],layersDepth[i], skybox, false);
        layerColor = max(layerColor, 2.0f * layerStep * color);

        vec4 bloom = (i==0 ? reflectionProbability : 1.0f) * accumulateColor(beginCoords,endCoords,layerStep,layersBloomSampler[i],layersDepth[i], skybox, false);
        layerBloom = max(layerBloom, 2.0f * layerStep * bloom);

        findRefr(i,nbegin,beginStartPos,beginCoords);
        findRefr(i,nend,endStartPos,endCoords);
    }

    outColor = accumulateColor(beginCoords,endCoords,step, Sampler, depth, skybox, true);
    outColor = max(layerColor, step * outColor);

    outBloom = accumulateColor(beginCoords,endCoords,step, bloomSampler, depth, skyboxBloom, true);
    outBloom = max(layerBloom, step * outBloom);
}
