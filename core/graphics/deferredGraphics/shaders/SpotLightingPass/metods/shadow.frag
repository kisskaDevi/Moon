float shadowFactor(sampler2D Sampler, vec4 coordinates)
{
    float result = 0.0f;

    vec3 lightSpaceNDC = coordinates.xyz / coordinates.w;
    vec2 coord = lightSpaceNDC.xy * 0.5f + 0.5f;

        int n = 8; int maxNoise = 1;
        float dang = 2.0f*pi/n; float dnoise = 0.001f;
        for(int j=0;j<n;j++)
        {
            for(float noise = dnoise; noise<dnoise*(maxNoise+1); noise+=dnoise)
            {
                vec2 dx = vec2(noise*cos(j*dang), noise*sin(j*dang));
                result += lightSpaceNDC.z-texture(Sampler, coord.xy + dx).x > 0.001f ? 1.0f : 0.0f;
            }
        }

     result /= maxNoise*n;

    return 1.0f - result;
}