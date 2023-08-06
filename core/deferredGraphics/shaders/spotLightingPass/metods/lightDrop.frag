float lightDrop(float distance)
{
    return pow(distance,1);
}

float lightDistribusion(const in vec3 position, const in vec3 lightPosition, const in mat4 lightProjMatrix, const in vec3 lightDirection){
    float fov = 2*atan(-1.0f/lightProjMatrix[1][1]);
    float theta = acos(dot(normalize(position.xyz - lightPosition),lightDirection));
    float arg = 3.141592653589793f * theta/fov;
    return pow(cos(arg),4);
}
