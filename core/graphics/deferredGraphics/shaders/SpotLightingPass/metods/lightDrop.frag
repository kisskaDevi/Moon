float lightDrop(float distance)
{
    float C = 1.0f;
    float L = 0.0866f*exp(-0.00144f*distance);
    float Q = 0.0283f*exp(-0.00289f*distance);

    return C + L * distance + Q * distance * distance;
}

float lightDistribusion(const in vec3 position, const in vec3 lightPosition, const in mat4 lightProjMatrix, const in vec3 lightDirection){
    float fov = asin(1.0f/sqrt(1.0f + lightProjMatrix[1][1] * lightProjMatrix[1][1]));
    float theta = acos(dot(normalize(position.xyz - lightPosition),lightDirection));
    float arg = 3.1415f/2.0f * theta/fov;
    return pow(cos(theta),16);
}
