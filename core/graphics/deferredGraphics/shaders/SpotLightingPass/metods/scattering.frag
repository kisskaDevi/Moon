float maxDist = 1e5f;

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
struct intersectionOutput{
    bool intersectionCondition;
    bool inside;
    float intersectionPoint1;
    float intersectionPoint2;
};
intersectionOutput findConeIntersection(const in vec4 viewPosition, const in vec4 viewDirection, const in mat4 lightProjMatrix, const in mat4 lightViewMatrix){
    intersectionOutput outputStatus;

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

    outputStatus.inside = pp < 0.0f;
    outputStatus.intersectionCondition = D > 0;
    outputStatus.intersectionPoint1 = outputStatus.intersectionCondition ? (-dp + sqrt(D))/dd : 0.0f;
    outputStatus.intersectionPoint2 = outputStatus.intersectionCondition ? (-dp - sqrt(D))/dd : 0.0f;

    return outputStatus;
}
float findFirstSolution(const in vec3 a, const in vec3 b, const in vec3 c, const in vec3 d){
    vec3 bc = cross(b,c);
    vec3 dcxbc = cross(cross(d,c),bc);
    vec3 acxbc = cross(cross(a,c),bc);

    return dot(dcxbc,acxbc)/dot(acxbc,acxbc);
}
float findTriangleIntersection(const in vec3 p0, const in vec3 d, const in vec3 v0, const in vec3 v1, const in vec3 v2){
    vec3 P0 = p0 - v2;
    vec3 V0 = v0 - v2;
    vec3 V1 = v1 - v2;

    float s0 = findFirstSolution(V0,V1,-d,P0);
    float s1 = findFirstSolution(V1,V0,-d,P0);
    float t  = findFirstSolution(-d,V1,V0,P0);

    return (s1<0.0f||s0+s1>1.0) ? 0.0f : t;
}
intersectionOutput findPyramidIntersection(const in vec4 viewPosition, const in vec4 viewDirection, const in vec4 lightPosition, const in mat4 lightProjMatrix, const in mat4 lightViewMatrix){
    intersectionOutput outputStatus;

    vec4 positionInLightCoord = lightProjMatrix * lightViewMatrix * viewPosition;
    positionInLightCoord /= positionInLightCoord.w;

    vec3 n = - normalize(vec3(lightViewMatrix[0][2],lightViewMatrix[1][2],lightViewMatrix[2][2]));
    vec3 u =   normalize(vec3(lightViewMatrix[0][0],lightViewMatrix[1][0],lightViewMatrix[2][0]));
    vec3 v =   normalize(vec3(lightViewMatrix[0][1],lightViewMatrix[1][1],lightViewMatrix[2][1]));

    float far  = -lightProjMatrix[3][2]/(-lightProjMatrix[2][2]-1.0f);
    float h = far/lightProjMatrix[1][1];
    float w = lightProjMatrix[1][1]/lightProjMatrix[0][0]*h;

    vec3 v0 = lightPosition.xyz;
    vec3 v1 = lightPosition.xyz + far*n + w*u + h*v;
    vec3 v2 = lightPosition.xyz + far*n + w*u - h*v;
    vec3 v3 = lightPosition.xyz + far*n - w*u + h*v;
    vec3 v4 = lightPosition.xyz + far*n - w*u - h*v;

    float t[4] = float[4](
        findTriangleIntersection(viewPosition.xyz,viewDirection.xyz,v0,v1,v2),
        findTriangleIntersection(viewPosition.xyz,viewDirection.xyz,v0,v2,v4),
        findTriangleIntersection(viewPosition.xyz,viewDirection.xyz,v0,v4,v3),
        findTriangleIntersection(viewPosition.xyz,viewDirection.xyz,v0,v3,v1)
    );

    outputStatus.inside = (abs(positionInLightCoord.x)<=1.0f)&&(abs(positionInLightCoord.y)<=1.0f)&&(positionInLightCoord.z>=0.0f);
    outputStatus.intersectionCondition = t[0]+t[1]+t[2]+t[3]!=0.0f;

    float t1 = 0.0f;
    float t2 = 0.0f;
    for(int i=0;i<4;i++){
        if(t[i]  !=0.0f) if(t1==0.0f) t1 = t[i];
        if(t[3-i]!=0.0f) if(t2==0.0f) t2 = t[3-i];
    }

    outputStatus.intersectionPoint1 = t1;
    if(dot(normalize(viewPosition.xyz - lightPosition.xyz),n)<=0.0f && outputStatus.inside){
        outputStatus.intersectionPoint2 = t2;
    }else{
        outputStatus.intersectionPoint2 = t2!=t1 ? t2 : 0.0f;
    }

    return outputStatus;
}
vec3 findLightDirection(const in mat4 lightViewMatrix){
    return - normalize(vec3(lightViewMatrix[0][2],lightViewMatrix[1][2],lightViewMatrix[2][2]));
}
float findFarPlane(const in mat4 lightProjMatrix){
    return lightProjMatrix[3][2]/(1.0f+lightProjMatrix[2][2]);
}
bool isPositionBehind(const in vec4 viewPosition, const in vec4 lightPosition, const in vec3 lightDirection){
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
    return atan(1.0f/lightProjMatrix[1][1]);
}
vec4 findPointColor(const float type, const in vec3 point, sampler2D lightTexture, const in vec2 lightCoordinates, const in vec4 lightColor, const in mat4 lightProjMatrix, const in vec3 lightPosition, const in vec3 lightDirection){
    float drop = lightDrop(length(lightPosition - point));
    float distribusion = lightDistribusion(point,lightPosition,lightProjMatrix,lightDirection);

    if(type == 0.0f){
        return max(texture(lightTexture, lightCoordinates.xy), lightColor)/drop*distribusion;
    }else{
        return max(texture(lightTexture, lightCoordinates.xy), lightColor)/drop/drop;
    }
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

    intersectionOutput outputStatus = type == 0.0f ? findConeIntersection(position,direction,lightProjMatrix,lightViewMatrix) :
                                                     findPyramidIntersection(position,direction,lightPosition,lightProjMatrix,lightViewMatrix);

    if(outputStatus.intersectionCondition){
        vec3 lightDirection = findLightDirection(lightViewMatrix);
        bool positionCondition = isPositionBehind(position,lightPosition,lightDirection);

        float tBegin = findBegin(outputStatus.intersectionPoint1,outputStatus.intersectionPoint2,outputStatus.inside,positionCondition);
        float tEnd = findEnd(outputStatus.intersectionPoint1,outputStatus.intersectionPoint2,outputStatus.inside,positionCondition);
        float dphi = 2.0f*findFov(lightProjMatrix)/(steps);

        for(int i=0;(i<steps);i++){
            float t = findPropagationFactor(i * dphi, position + direction * tBegin, direction, lightPosition);
            vec4 pointOfScattering = position + direction * (tBegin + t);
            if((depthMap - findDepth(projView,pointOfScattering) > 0.0f) && (tBegin + t < tEnd) && (t > 0.0f)){
                vec3 coordinates = coordinatesInLocalBasis(lightProjViewMatrix,vec4(pointOfScattering.xyz,1.0f));
                if(coordinates.z<texture(shadowMap, coordinates.xy).x){
                    outScatteringColor += findPointColor(type, pointOfScattering.xyz,lightTexture,coordinates.xy,lightColor,lightProjMatrix,lightPosition.xyz,lightDirection);
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

    intersectionOutput outputStatus = type == 0.0f ? findConeIntersection(position,direction,lightProjMatrix,lightViewMatrix) :
                                                     findPyramidIntersection(position,direction,lightPosition,lightProjMatrix,lightViewMatrix);

    if(outputStatus.intersectionCondition){
        vec3 lightDirection = findLightDirection(lightViewMatrix);
        bool positionCondition = isPositionBehind(position,lightPosition,lightDirection);

        float tBegin = findBegin(outputStatus.intersectionPoint1,outputStatus.intersectionPoint2,outputStatus.inside,positionCondition);
        float tEnd = findEnd(outputStatus.intersectionPoint1,outputStatus.intersectionPoint2,outputStatus.inside,positionCondition);
        float dphi = 2.0f*findFov(lightProjMatrix)/(steps);

        for(int i=0;(i<steps);i++){
            float t = findPropagationFactor(i * dphi, position + direction * tBegin, direction, lightPosition);
            vec4 pointOfScattering = position + direction * (tBegin + t);
            if((depthMap - findDepth(projView,pointOfScattering) > 0.0f) && (tBegin + t < tEnd) && (t > 0.0f)){
                vec3 coordinates = coordinatesInLocalBasis(lightProjViewMatrix,vec4(pointOfScattering.xyz,1.0f));
                outScatteringColor += findPointColor(type,pointOfScattering.xyz,lightTexture,coordinates.xy,lightColor,lightProjMatrix,lightPosition.xyz,lightDirection);
            }
        }
    }

    return outScatteringColor/steps;
}
