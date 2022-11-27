float maxDist = 1000.0f;

float conicDot(const in vec4 v1, const in vec4 v2){
    return v1.x * v2.x + v1.y * v2.y - v1.z * v2.z;
}
float findMinInter(const float t1, const float t2){
    return min((t1 > 0.0f ? t1 : maxDist), (t2 > 0.0f ? t2 : maxDist));
}
float findMaxInter(const float t1, const float t2){
    return max((t1 >0.0f ? t1 : maxDist), (t2 >0.0f ? t2 : maxDist));
}
float findBegin(const float t1, const float t2, const bool insideCondition, const bool positionCondition){
    return insideCondition ? ( positionCondition ? 0.0f : findMaxInter(t1,t2) ) : findMinInter(t1,t2);
}
float findEnd(const float t1, const float t2, const bool insideCondition, const bool positionCondition){
    return insideCondition ? ( positionCondition ? findMinInter(t1,t2) : maxDist ) : findMaxInter(t1,t2);
}
struct intersectionConeOutput{
    bool intersectionCondition;
    bool insideCone;
    float intersectionPoint1;
    float intersectionPoint2;
};
intersectionConeOutput findIntersection(const in vec4 viewPosition, const in vec4 viewDirection, const in mat4 lightProjMatrix, const in mat4 lightViewMatrix){
    intersectionConeOutput outputStatus;

    float far = lightProjMatrix[3][2]/(1.0f+lightProjMatrix[2][2]);
    float height = far/lightProjMatrix[1][1];
    float width = far/lightProjMatrix[0][0];

    float xAxis = width/far;
    float yAxis = height/far;

    vec4 directionInLightCoord = lightViewMatrix * viewDirection;
    vec4 positionInLightCoord = lightViewMatrix * viewPosition;

    directionInLightCoord = vec4(directionInLightCoord.x/xAxis,directionInLightCoord.y/yAxis,directionInLightCoord.z,directionInLightCoord.w);
    positionInLightCoord = vec4(positionInLightCoord.x/xAxis,positionInLightCoord.y/yAxis,positionInLightCoord.z,positionInLightCoord.w);

    float dp = conicDot(positionInLightCoord, directionInLightCoord);
    float dd = conicDot(directionInLightCoord, directionInLightCoord);
    float pp = conicDot(positionInLightCoord, positionInLightCoord);

    float D = dp*dp - dd*pp;

    outputStatus.insideCone = pp < 0.0f;
    outputStatus.intersectionCondition = D > 0;
    outputStatus.intersectionPoint1 = outputStatus.intersectionCondition ? (-dp + sqrt(D))/dd : 0.0f;
    outputStatus.intersectionPoint2 = outputStatus.intersectionCondition ? (-dp - sqrt(D))/dd : 0.0f;

    return outputStatus;
}
vec3 findLightDirection(const in mat4 lightViewMatrix){
    return - normalize(vec3(lightViewMatrix[0][2],lightViewMatrix[1][2],lightViewMatrix[2][2]));
}
float findFarPlane(const in mat4 lightProjMatrix){
    return lightProjMatrix[3][2]/(1.0f+lightProjMatrix[2][2]);
}
bool isPositionBehindCone(const in vec4 viewPosition, const in vec4 lightPosition, const in vec3 lightDirection){
    return dot(normalize(viewPosition.xyz - lightPosition.xyz),lightDirection) > 0.0f;
}
float findDepth(const in mat4 projView, vec4 position){
    float cameraViewZ = projView[0][2]*position.x + projView[1][2]*position.y + projView[2][2]*position.z + projView[3][2];
    float cameraViewW = projView[0][3]*position.x + projView[1][3]*position.y + projView[2][3]*position.z + projView[3][3];
    return cameraViewZ/cameraViewW;
}
vec3 coordinatesInLocalBasis(const in mat4 projViewMatrix, vec4 position){
    vec4 projection = projViewMatrix * position;
    vec3 normProjection = (projection.xyz)/projection.w;
    vec2 coordinatesXY = normProjection.xy * 0.5f + 0.5f;
    return vec3(coordinatesXY,normProjection.z);
}
float findFov(const in mat4 lightProjMatrix){
    return asin(1.0f/sqrt(1.0f + lightProjMatrix[1][1] * lightProjMatrix[1][1]));
}
vec4 findPointColor(const in vec3 point, sampler2D lightTexture, const in vec2 lightCoordinates, const in vec4 lightColor, const in mat4 lightProjMatrix, const in vec3 lightPosition, const in vec3 lightDirection){
    float drop = lightDrop(length(lightPosition - point));
    float distribusion = lightDistribusion(point,lightPosition,lightProjMatrix,lightDirection);

    return max(texture(lightTexture, lightCoordinates.xy), lightColor)/drop*distribusion;
}
float findPropagationFactor(float phi, const in vec4 position, const in vec4 direction, const in vec4 lightPosition){
    vec3 y = position.xyz - lightPosition.xyz;
    vec3 x = direction.xyz;
    float xy = dot(x,y);
    float xx = dot(x,x);
    float yy = dot(y,y);
    float a = xy*xy - xx*yy*cos(phi);
    float b = 2.0f*xy*yy*(1.0f - cos(phi));
    float c = yy*yy*(1.0f - cos(phi));

    return (-b - sqrt(b*b-4.0f*a*c))/(2.0f*a);
}

vec4 LightScattering(
        const int steps,
        const in mat4 lightViewMatrix,
        const in mat4 lightProjMatrix,
        const in mat4 lightProjViewMatrix,
        const in vec4 lightPosition,
        const in vec4 lightColor,
        const in mat4 projView,
        const in vec4 position,
        const in vec4 fragPosition,
        sampler2D lightTexture,
        sampler2D shadowMap,
        float depthMap,
        float type)
{
    vec4 outScatteringColor = vec4(0.0f);

    vec4 direction = normalize(fragPosition - position);
    intersectionConeOutput outputStatus = findIntersection(position,direction,lightProjMatrix,lightViewMatrix);

    if(outputStatus.intersectionCondition){
        vec3 lightDirection = findLightDirection(lightViewMatrix);
        bool positionCondition = isPositionBehindCone(position,lightPosition,lightDirection);

        float tBegin = findBegin(outputStatus.intersectionPoint1,outputStatus.intersectionPoint2,outputStatus.insideCone,positionCondition);
        float tEnd = findEnd(outputStatus.intersectionPoint1,outputStatus.intersectionPoint2,outputStatus.insideCone,positionCondition);
        float dphi = 2.0f*findFov(lightProjMatrix)/(steps);

        for(int i=0;(i<steps);i++){
            float t = findPropagationFactor(i * dphi, position + direction * tBegin, direction, lightPosition);
            vec4 pointOfScattering = position + direction * (tBegin + t);
            if((depthMap - findDepth(projView,pointOfScattering) > 0.0f) && (tBegin + t < tEnd) && (t > 0.0f)){
                vec3 coordinates = coordinatesInLocalBasis(lightProjViewMatrix,vec4(pointOfScattering.xyz,1.0f));
                if(coordinates.z<texture(shadowMap, coordinates.xy).x){
                    outScatteringColor += findPointColor(pointOfScattering.xyz,lightTexture,coordinates.xy,lightColor,lightProjMatrix,lightPosition.xyz,lightDirection);
                }
            }
        }
    }

    return outScatteringColor/steps;
}

vec4 LightScattering(
        const int steps,
        const in mat4 lightViewMatrix,
        const in mat4 lightProjMatrix,
        const in mat4 lightProjViewMatrix,
        const in vec4 lightPosition,
        const in vec4 lightColor,
        const in mat4 projView,
        const in vec4 position,
        const in vec4 fragPosition,
        sampler2D lightTexture,
        float depthMap,
        float type)
{
    vec4 outScatteringColor = vec4(0.0f);

    vec4 direction = normalize(fragPosition - position);
    intersectionConeOutput outputStatus = findIntersection(position,direction,lightProjMatrix,lightViewMatrix);

    if(outputStatus.intersectionCondition){
        vec3 lightDirection = findLightDirection(lightViewMatrix);
        bool positionCondition = isPositionBehindCone(position,lightPosition,lightDirection);

        float tBegin = findBegin(outputStatus.intersectionPoint1,outputStatus.intersectionPoint2,outputStatus.insideCone,positionCondition);
        float tEnd = findEnd(outputStatus.intersectionPoint1,outputStatus.intersectionPoint2,outputStatus.insideCone,positionCondition);
        float dphi = 2.0f*findFov(lightProjMatrix)/(steps);

        for(int i=0;(i<steps);i++){
            float t = findPropagationFactor(i * dphi, position + direction * tBegin, direction, lightPosition);
            vec4 pointOfScattering = position + direction * (tBegin + t);
            if((depthMap - findDepth(projView,pointOfScattering) > 0.0f) && (tBegin + t < tEnd) && (t > 0.0f)){
                vec3 coordinates = coordinatesInLocalBasis(lightProjViewMatrix,vec4(pointOfScattering.xyz,1.0f));
                outScatteringColor += findPointColor(pointOfScattering.xyz,lightTexture,coordinates.xy,lightColor,lightProjMatrix,lightPosition.xyz,lightDirection);
            }
        }
    }

    return outScatteringColor/steps;
}
