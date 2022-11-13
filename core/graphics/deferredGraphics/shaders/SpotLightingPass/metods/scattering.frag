vec4 LightScattering(
        int steps,
        mat4 lightProjView,
        mat4 projview,
        vec4 eyePosition,
        vec4 glPosition,
        vec4 lightColor,
        vec3 lightPosition,
        sampler2D shadowMap,
        sampler2D lightTexture,
        float depthMap,
        float type)
{
    vec4 outScatteringColor = vec4(0.0f,0.0f,0.0f,0.0f);
    vec3 eyeDirection = glPosition.xyz - eyePosition.xyz;
    vec3 dstep = eyeDirection/steps;

    int insideCounter = 0;

    for(int step=1;step<=steps;step++){
        vec3 pointOfScattering = eyePosition.xyz + step * dstep;
        float cameraViewZ = projview[0][2]*pointOfScattering.x + projview[1][2]*pointOfScattering.y + projview[2][2]*pointOfScattering.z + projview[3][2];
        float cameraViewW = projview[0][3]*pointOfScattering.x + projview[1][3]*pointOfScattering.y + projview[2][3]*pointOfScattering.z + projview[3][3];
        if(depthMap>cameraViewZ/cameraViewW){
            vec4 lightView = lightProjView * vec4(pointOfScattering, 1.0f);
            vec3 lightSpaceNDC = lightView.xyz/lightView.w;
            if(!outsideSpotCondition(lightSpaceNDC,type))
            {
                vec2 coordinates = lightSpaceNDC.xy * 0.5f + 0.5f;
                if(lightSpaceNDC.z<texture(shadowMap, coordinates.xy).x)
                {
                    vec4 color = texture(lightTexture, coordinates.xy);
                    color = vec4(max(color.x,lightColor.x),max(color.y,lightColor.y),max(color.z,lightColor.z),max(color.a,lightColor.a));

                    float len = length(lightPosition - pointOfScattering);

                    outScatteringColor += vec4(color.xyz,0.0f)*exp(-0.2f*(len));
                    insideCounter++;
                }
            }
        }
    }
    outScatteringColor /= insideCounter*4;
    return outScatteringColor;
}

vec4 LightScattering(
        int steps,
        mat4 lightProjView,
        mat4 projview,
        vec4 eyePosition,
        vec4 glPosition,
        vec4 lightColor,
        vec3 lightPosition,
        sampler2D lightTexture,
        float depthMap,
        float type)
{
    vec4 outScatteringColor = vec4(0.0f,0.0f,0.0f,0.0f);
    vec3 eyeDirection = glPosition.xyz - eyePosition.xyz;
    vec3 dstep = eyeDirection/steps;

    int insideCounter = 0;

    for(int step=1;step<=steps;step++){
        vec3 pointOfScattering = eyePosition.xyz + step * dstep;
        float cameraViewZ = projview[0][2]*pointOfScattering.x + projview[1][2]*pointOfScattering.y + projview[2][2]*pointOfScattering.z + projview[3][2];
        float cameraViewW = projview[0][3]*pointOfScattering.x + projview[1][3]*pointOfScattering.y + projview[2][3]*pointOfScattering.z + projview[3][3];
        if(depthMap>cameraViewZ/cameraViewW){
            vec4 lightView = lightProjView * vec4(pointOfScattering, 1.0f);
            vec3 lightSpaceNDC = lightView.xyz/lightView.w;
            if(!outsideSpotCondition(lightSpaceNDC,type))
            {
                vec2 coordinates = lightSpaceNDC.xy * 0.5f + 0.5f;
                vec4 color = texture(lightTexture, coordinates.xy);
                color = vec4(max(color.x,lightColor.x),max(color.y,lightColor.y),max(color.z,lightColor.z),max(color.a,lightColor.a));

                float len = length(lightPosition - pointOfScattering);

                outScatteringColor += vec4(color.xyz,0.0f)*exp(-0.2f*(len));
                insideCounter++;
            }
        }
    }
    outScatteringColor /= insideCounter*4;
    return outScatteringColor;
}
