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
layout(set = 0, binding = 2) uniform sampler2D position;
layout(set = 0, binding = 3) uniform sampler2D normal;
layout(set = 0, binding = 4) uniform sampler2D depth;
layout(set = 0, binding = 5) uniform sampler2D layersSampler[transparentLayersCount];
layout(set = 0, binding = 6) uniform sampler2D layersPosition[transparentLayersCount];
layout(set = 0, binding = 7) uniform sampler2D layersNormal[transparentLayersCount];
layout(set = 0, binding = 8) uniform sampler2D layersDepth[transparentLayersCount];

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

mat4 projview = global.proj * global.view;

float h = 0.01f;
vec3 n = vec3(1.33f, 1.33f + 1.0f, 1.33f + 2.0f);

vec3 getRefrCoords(const in vec3 layerPointPosition, const in vec3 layerPointNormal, float n){
    vec3 beamDirection = normalize(layerPointPosition - global.eyePosition.xyz);
    float cosAlpha = - dot(layerPointNormal,beamDirection);
    float sinAlpha = sqrt(1.0f - cosAlpha * cosAlpha);

    float deviation = h * sinAlpha * (1.0f - cosAlpha / sqrt(n*n - sinAlpha*sinAlpha));

    vec3 direction = layerPointNormal * dot(beamDirection,beamDirection) - beamDirection * dot(beamDirection,layerPointNormal);

    vec4 dpos = vec4(layerPointPosition + deviation.r * direction,1.0f);

    vec4 dposProj = projview * dpos;
    return vec3(dposProj.xy/dposProj.w * 0.5f + 0.5f,dposProj.z/dposProj.w);
}

vec3 layerPointPosition(const in int i, const in vec2 coord){
    if(i<transparentLayersCount){
        return texture(layersPosition[i], coord).xyz;
    }else{
        return texture(position, coord).xyz;
    }
}

vec3 layerPointNormal(const in int i, const in vec2 coord){
    if(i<transparentLayersCount){
        return normalize(texture(layersNormal[i],coord).xyz);
    }else{
        return normalize(texture(normal,coord).xyz);
    }
}

vec4 layerColor(const in int i, const in vec2 coord){
    if(i<transparentLayersCount){
        return texture(layersSampler[0],coord);
    }else{
        return texture(Sampler,coord);
    }
}

bool outsideCond(const in vec2 coords){
    return coords.x<1.0f&&coords.y<1.0f&&coords.x>0.0f&&coords.y>0.0f;
}

bool depthCond(float z, vec2 coords){
    return z < texture(depth,coords.xy).r;
}

vec4 getRefrColor(const int i, inout vec3 coords, const float n)
{
    if(depthCond(texture(layersDepth[i],coords.xy).r,coords.xy))
    {
        coords  = getRefrCoords(layerPointPosition(i,coords.xy), layerPointNormal(i,coords.xy),n);
        return outsideCond(coords.xy) && depthCond(coords.z,coords.xy) ? layerColor(transparentLayersCount,coords.xy) : vec4(0.0f);
    }else{
        return vec4(0.0f);
    }
}

void main()
{
    vec4 layerColor0 = depthCond(texture(layersDepth[0],fragTexCoord).r,fragTexCoord) ? layerColor(0,fragTexCoord) : vec4(0.0f);
    outColor = (layerColor0.r+layerColor0.g+layerColor0.b>0.0f) ? layerColor0 : texture(Sampler,fragTexCoord);

    vec3 coordsR = vec3(fragTexCoord,0.0f);
    vec3 coordsG = vec3(fragTexCoord,0.0f);
    vec3 coordsB = vec3(fragTexCoord,0.0f);

    vec4 refrColor;
    for(int i=0;i<transparentLayersCount;i++)
    {
        refrColor = vec4(getRefrColor(i,coordsR,n.r).r, getRefrColor(i,coordsG,n.g).g, getRefrColor(i,coordsB,n.b).b, 0.0f);

        if(refrColor.r+refrColor.g+refrColor.b>0.0f)
            outColor = layerColor(0,fragTexCoord) + refrColor;
    }
}
