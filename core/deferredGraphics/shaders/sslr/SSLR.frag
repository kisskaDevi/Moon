#version 450

#include "../__methods__/defines.glsl"
#include "../__methods__/pbr.glsl"

layout(set = 0, binding = 0) uniform GlobalUniformBuffer {
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

vec4 pointOfView = vec4(global.eyePosition.xyz, 1.0);
mat4 projview = global.proj * global.view;

vec4 findPositionInPlane(const in mat4 projview, const in vec4 position) {
    vec4 positionProj = projview * position;
    return positionProj / positionProj.w;
}

vec2 findIncrement(const in mat4 projview, const in vec4 position, const in vec4 direction) {
    vec4 start = findPositionInPlane(projview, position);
    vec4 end = findPositionInPlane(projview, position + direction);
    vec2 planeDir = (end.xy - start.xy);
    return planeDir;
}

bool insideCondition(const in vec2 coords) {
    return coords.x > 0.0 && coords.y > 0.0 && coords.x < 1.0 && coords.y < 1.0;
}

vec4 findSampler(const in vec2 coords, bool depthCondition) {
    return depthCondition ? texture(Sampler, coords) : texture(layerSampler, coords);
}

vec4 findPosition(const in vec2 coords, bool depthCondition) {
    return depthCondition ? vec4(texture(position, coords).xyz, 1.0) : vec4(texture(layerPosition, coords).xyz, 1.0);
}

vec4 findNormal(const in vec2 coords, bool depthCondition) {
    return depthCondition ? vec4(texture(normal, coords).xyz, 0.0) : vec4(texture(layerNormal, coords).xyz, 0.0);
}

vec4 SSLR(int steps, float incrementFactor, float resolution) {
    vec4 SSLR = vec4(0.0);

    bool depthCond = texture(depth, fragTexCoord).r < texture(layerDepth, fragTexCoord).r;
    vec4 pointPosition = findPosition(fragTexCoord, depthCond);
    vec4 pointNormal = vec4(0.0);

    vec2 offset = 1.0 / textureSize(normal, 0);
    for(int i = -1; i < 2; i++) {
        for(int j = -1; j < 2; j++) {
            pointNormal += findNormal(fragTexCoord + offset * vec2(i, j), depthCond);
        }
    }
    pointNormal /= 9.0;

    vec4 reflectDirection = normalize(reflect(pointPosition - pointOfView, pointNormal));
    vec2 increment = incrementFactor * findIncrement(projview, pointPosition, reflectDirection);
    vec2 planeCoords = findPositionInPlane(projview, pointPosition).xy * vec2(0.5) + vec2(0.5);

    for(int i = 0; i < steps && insideCondition(planeCoords); i++, planeCoords += increment) {
        bool depthCond = texture(depth, planeCoords).r - texture(layerDepth, planeCoords).r < 0.0;
        vec4 direction = normalize(findPosition(planeCoords, depthCond) - pointPosition);

        float cosTheta = dot(reflectDirection, direction);
        float cosPhi = dot(findNormal(planeCoords, depthCond), direction);

        if((cosTheta >= resolution) && (cosPhi <= 0.0)) {
            SSLR = findSampler(planeCoords, depthCond);
        }
    }

    return SSLR;
}

void main() {
    outColor = vec4(0.0);

    float roug = 1.0 - texture(position, fragTexCoord).a;
    //outColor += SSLR(20, 0.2f, 0.9995);
}
