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

// The following equation models the Fresnel reflectance term of the spec equation (aka F())
// Implementation of fresnel from [4], Equation 15
vec3 specularReflection(vec3 specularEnvironmentR0, vec3 specularEnvironmentR90, float VdotH)
{
        return specularEnvironmentR0 + (specularEnvironmentR90 - specularEnvironmentR0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
}

// This calculates the specular geometric attenuation (aka G()),
// where rougher material will reflect less light back to the viewer.
// This implementation is based on [1] Equation 4, and we adopt their modifications to
// alphaRoughness as input as originally proposed in [2].
float geometricOcclusion(float NdotL, float NdotV, float r)
{
        float attenuationL = 2.0 * NdotL / (NdotL + sqrt(r * r + (1.0 - r * r) * (NdotL * NdotL)));
        float attenuationV = 2.0 * NdotV / (NdotV + sqrt(r * r + (1.0 - r * r) * (NdotV * NdotV)));
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

// Basic Lambertian diffuse
// Implementation from Lambert's Photometria https://archive.org/details/lambertsphotome00lambgoog
// See also [1], Equation 1
vec3 diffuse(vec3 diffuseColor)
{
    return diffuseColor / pi;
}

vec4 PBR(   vec4 position,
            vec4 normal,
            vec4 baseColorTexture,
            vec4 eyePosition,
            vec4 lightColor,
            vec3 lightPosition)
{
        vec3 f0 = vec3(0.04);

        float metallic = normal.a;
        float perceptualRoughness = position.a;

        vec4 baseColor = SRGBtoLINEAR(baseColorTexture);
        vec3 diffuseColor = baseColor.rgb * (vec3(1.0) - f0);
        diffuseColor *= 1.0 - metallic;

        float alphaRoughness = perceptualRoughness * perceptualRoughness;

        vec3 specularColor = mix(f0, baseColor.rgb, metallic);

        float reflectance = max(max(specularColor.r, specularColor.g), specularColor.b);
        float reflectance90 = clamp(reflectance * 25.0, 0.0, 1.0);
        vec3 specularEnvironmentR0 = specularColor.rgb;
        vec3 specularEnvironmentR90 = vec3(1.0, 1.0, 1.0) * reflectance90;

        vec3 eyeDirection   = normalize(eyePosition.xyz - position.xyz);
        vec3 lightDirection = normalize(lightPosition - position.xyz);
        vec3 Normal	    = normal.xyz;
        vec3 reflect	    = -normalize(reflect(eyeDirection, Normal));
        vec3 H		    = normalize(eyeDirection + lightDirection);

        vec3  F = specularReflection(specularEnvironmentR0, specularEnvironmentR90, clamp(dot(eyeDirection, H), 0.0, 1.0));
        float G = geometricOcclusion(clamp(dot(Normal, lightDirection), 0.001, 1.0), clamp(abs(dot(Normal, eyeDirection)), 0.001, 1.0), alphaRoughness);
        float D = microfacetDistribution(clamp(dot(Normal, H), 0.0, 1.0), alphaRoughness);

        vec3 diffuseContrib = (1.0f - F) * diffuse(diffuseColor);
        vec3 specContrib = F * G * D / (4.0 * clamp(dot(Normal, lightDirection), 0.001, 1.0) * clamp(abs(dot(Normal, eyeDirection)), 0.001, 1.0));
        vec4 outColor = vec4( clamp(dot(Normal, lightDirection), 0.001, 1.0) * lightColor.xyz * (diffuseContrib + specContrib), baseColor.a);

    return outColor;
}