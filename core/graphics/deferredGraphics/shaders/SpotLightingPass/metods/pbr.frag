vec4 SRGBtoLINEAR(vec4 srgbIn)
{
        #ifdef MANUAL_SRGB
            #ifdef SRGB_FAST_APPROXIMATION
                vec3 linOut = pow(srgbIn.xyz,vec3(2.2));
            #else //SRGB_FAST_APPROXIMATION
                vec3 bLess = step(vec3(0.04045),srgbIn.xyz);
                vec3 linOut = mix( srgbIn.xyz/vec3(12.92), pow((srgbIn.xyz+vec3(0.055))/vec3(1.055),vec3(2.4)), bLess );
            #endif //SRGB_FAST_APPROXIMATION
            return vec4(linOut,srgbIn.w);
        #else //MANUAL_SRGB
        return srgbIn;
        #endif //MANUAL_SRGB
}

float geometricOcclusion(float NdotL, float NdotV, float k)
{
    float attenuationL = NdotL / (k + (1.0 - k) * (NdotL));
    float attenuationV = NdotV / (k + (1.0 - k) * (NdotV));
    return attenuationL * attenuationV;
}

// The following equation(s) model the distribution of microfacet normals across the area being drawn (aka D())
// Implementation from "Average Irregularity Representation of a Roughened Surface for Ray Reflection" by T. S. Trowbridge, and K. P. Reitz
// Follows the distribution function recommended in the SIGGRAPH 2013 course notes from EPIC Games [1], Equation 3.
float microfacetDistribution(float NdotH, float alphaRoughness)
{
        float roughnessSq = alphaRoughness * alphaRoughness;
        float f = (NdotH * roughnessSq - NdotH) * NdotH + 1.0;
        return roughnessSq / (pi * f * f);
}

vec3 diffuse(const in vec4 BaseColor, const in float metallic, const in vec3 f0){
    vec3 diffuseColor = BaseColor.rgb * (vec3(1.0) - f0);
    diffuseColor *= 1.0 - metallic;

    return diffuseColor / pi;
}

vec3 specularReflection(vec3 specular, float DdotN){
    return specular + (vec3(1.0f) - specular) * pow(1.0f - DdotN, 5);
}

vec4 PBR(   vec4 position,
            vec4 normal,
            vec4 baseColorTexture,
            vec4 eyePosition,
            vec4 lightColor,
            vec3 lightPosition)
{
    vec3 Direction      = normalize(eyePosition.xyz - position.xyz);
    vec3 LightDirection = normalize(lightPosition - position.xyz);
    vec3 Normal         = normal.xyz;
    vec3 H              = normalize(Direction + LightDirection);
    vec4 BaseColor      = SRGBtoLINEAR(baseColorTexture);

    float metallic = normal.a;
    float perceptualRoughness = position.a;

    vec3 f0 = vec3(0.04);

    vec3 specularColor  = mix(f0, BaseColor.rgb, metallic);
    vec3 F              = 0.1 * specularReflection(specularColor, clamp(dot(Direction, Normal), 0.0, 1.0));

    float alphaRoughness = (perceptualRoughness + 1) * (perceptualRoughness + 1) / 8.0f;
    float G = geometricOcclusion(clamp(dot(Normal, LightDirection), 0.001, 1.0), clamp(abs(dot(Normal, Direction)), 0.001, 1.0), alphaRoughness);
    float D = microfacetDistribution(clamp(dot(Normal, H), 0.0, 1.0), perceptualRoughness);

    vec3 diffuseContrib = (1.0f - F) * diffuse(BaseColor, metallic, f0);
    vec3 specContrib = F * G * D / (4.0 * clamp(dot(Normal, LightDirection), 0.001, 1.0) * clamp(abs(dot(Normal, Direction)), 0.001, 1.0));

    vec4 outColor = vec4(clamp(dot(Normal, LightDirection), 0.001, 1.0) * lightColor.xyz * (diffuseContrib + specContrib), BaseColor.a);

    return outColor;
}
