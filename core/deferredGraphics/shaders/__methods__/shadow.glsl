#ifndef SHADOW
#define SHADOW

float shadowFactor(sampler2D Sampler, vec4 coordinates) {
    float result = 0.0;

    vec3 lightSpaceNDC = coordinates.xyz / coordinates.w;
    vec2 coord = lightSpaceNDC.xy * 0.5 + 0.5;

    // int n = 8;
    // int maxNoise = 1;
    // float dang = 2.0 * pi / n;
    // float dnoise = 0.001;
    // for(int j = 0; j < n; j++) {
    //     for(float noise = dnoise; noise < dnoise * (maxNoise + 1); noise += dnoise) {
    //         vec2 dx = vec2(noise * cos(j * dang), noise * sin(j * dang));
    //         result += lightSpaceNDC.z - texture(Sampler, coord.xy + dx).x > 0.001 ? 1.0 : 0.0;
    //     }
    // }
    // return 1.0 - result / (maxNoise * n);

    // simple
    return lightSpaceNDC.z >= texture(Sampler, coord.xy).x ? 0.0f : 1.0f;
}

#endif