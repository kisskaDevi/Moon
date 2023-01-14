#version 450
#define pi 3.141592653589793f

layout(set = 0, binding = 0) uniform GlobalUniformBuffer
{
    mat4 view;
    mat4 proj;
    vec4 eyePosition;
} global;
layout(set = 0, binding = 1) uniform sampler2D position;
layout(set = 0, binding = 2) uniform sampler2D normal;
layout(set = 0, binding = 3) uniform sampler2D Sampler;
layout(set = 0, binding = 4) uniform sampler2D depth;
layout(set = 0, binding = 5) uniform sampler2D layerPosition;
layout(set = 0, binding = 6) uniform sampler2D layerNormal;
layout(set = 0, binding = 7) uniform sampler2D layerSampler;
layout(set = 0, binding = 8) uniform sampler2D layerDepth;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

vec4 pointOfView = vec4(global.eyePosition.xyz,1.0f);
mat4 projview = global.proj * global.view;

vec4 findPositionInPlane(const in mat4 projview, const in vec4 position){
    vec4 positionProj = projview * position;
    return positionProj/positionProj.w;
}

vec2 findIncrement(const in mat4 projview, const in vec4 position, const in vec4 direction){
    vec4 start = findPositionInPlane(projview,position);
    vec4 end = findPositionInPlane(projview,position + direction);
    vec2 planeDir = (end.xy - start.xy);
    return planeDir;
}

bool insideCondition(const in vec2 coords){
    return coords.x > 0.0f && coords.y > 0.0f && coords.x < 1.0f && coords.y < 1.0f;
}

vec4 findSampler(const in vec2 coords){
    return texture(depth, coords).r - texture(layerDepth, coords).r < 0.0 ? texture(Sampler, coords) : texture(layerSampler, coords);
}

vec4 findPosition(const in vec2 coords){
    return texture(depth, coords).r - texture(layerDepth, coords).r < 0.0 ? vec4(texture(position, coords).xyz,1.0f) : vec4(texture(layerPosition, coords).xyz,1.0f);
}

vec4 findNormal(const in vec2 coords){
    return texture(depth, coords).r - texture(layerDepth, coords).r < 0.0 ? vec4(texture(normal, coords).xyz,0.0f) : vec4(texture(layerNormal, coords).xyz,0.0f);
}

float findDepth(const in vec2 coords){
    return texture(depth, coords).r - texture(layerDepth, coords).r < 0.0 ? texture(depth, coords).r : texture(layerDepth, coords).r;
}

vec4 SSLR(int steps, float incrementFactor, float resolution)
{
    vec4 SSLR = vec4(0.0f);

    vec4 pointPosition = findPosition(fragTexCoord);
    vec4 pointNormal = findNormal(fragTexCoord);
    float pointDepth = findDepth(fragTexCoord);

    vec4 reflectDirection = normalize(reflect(pointPosition - pointOfView, pointNormal));
    vec2 increment = incrementFactor * findIncrement(projview,pointPosition,reflectDirection);
    vec2 planeCoords = findPositionInPlane(projview,pointPosition).xy * vec2(0.5) + vec2(0.5);

    for(int i = 0; i < steps; i++, planeCoords += increment){
        vec4 direction = normalize(findPosition(planeCoords) - pointPosition);

        float cosTheta = dot(reflectDirection, direction);
        float cosPhi = dot(direction,findNormal(planeCoords));
        if((1.0f - cosTheta <= resolution) && (cosPhi<0.0f)){
            SSLR += findSampler(planeCoords);
            break;
        }
    }

    return SSLR;
}

void main()
{
    outColor = vec4(0.0f);

    outColor += SSLR(100,0.05f,0.0001);
}
