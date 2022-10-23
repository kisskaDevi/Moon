#version 450
#define MANUAL_SRGB 1
#define pi 3.141592653589793f

layout (push_constant) uniform PC
{
    float minAmbientFactor;
}pc;

layout(location = 0)	in vec4 eyePosition;
layout(location = 1)	in vec2 fragTexCoord;
layout(location = 2)	in vec4 glPosition;

layout(location = 3)	flat in vec3 lightPosition;
layout(location = 4)	flat in vec4 lightColor;
layout(location = 5)	flat in vec4 lightProp;
layout(location = 6)	flat in mat4 lightProjView;
layout(location = 10)	flat in mat4 projview;

layout(input_attachment_index = 0, binding = 0) uniform subpassInput inPositionTexture;
layout(input_attachment_index = 1, binding = 1) uniform subpassInput inNormalTexture;
layout(input_attachment_index = 2, binding = 2) uniform subpassInput inBaseColorTexture;
layout(input_attachment_index = 3, binding = 3) uniform subpassInput inEmissiveTexture;
layout(input_attachment_index = 4, binding = 4) uniform subpassInput inDepthTexture;

layout(set = 1, binding = 1) uniform sampler2D shadowMap;
layout(set = 1, binding = 2) uniform sampler2D lightTexture;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outBlur;
layout(location = 2) out vec4 outBloom;

struct attenuation{
    float C;
    float L;
    float Q;
};

struct Vector{
    vec3 eyeDirection;
    vec3 lightDirection;
    vec3 normal;
    vec3 reflect;
    vec3 H;
}vector;

vec4 position;
vec4 normal;
vec4 baseColorTexture;
vec4 emissiveTexture;

vec4 fragLightPosition;
vec4 textureLightColor;

float type;
float lightPowerFactor;
float lightDropFactor;

//===================================================functions====================================================================//
bool		outsideSpotCondition(vec3 lightSpaceNDC, float type);
float		shadowFactor();
vec4		SRGBtoLINEAR(vec4 srgbIn);
vec3		specularReflection(vec3 specularEnvironmentR0, vec3 specularEnvironmentR90, float VdotH);
float		geometricOcclusion(float NdotL, float NdotV, float r);
float		microfacetDistribution(float NdotH, float alphaRoughness);
vec3		diffuse(vec3 diffuseColor);
attenuation	getK(float distance);

//===================================================outImages====================================================================//

vec4 PBR(vec4 outColor)
{
    //	    clamp(x,xmin,xmax) = min(max(x,xmin),xmax)
    //	    mix(vecx,vecy,a) = vecx + (vecy-vecx)*a

    fragLightPosition = lightProjView * vec4(position.xyz,1.0f);
    textureLightColor = vec4(0.0f,0.0f,0.0f,1.0f);

    float metallic = subpassLoad(inNormalTexture).a;
    float perceptualRoughness = subpassLoad(inPositionTexture).a;
    vec3 diffuseColor;
    vec4 baseColor = baseColorTexture;
    baseColor = SRGBtoLINEAR(baseColor);

    vec3 f0 = vec3(0.04);

    diffuseColor = baseColor.rgb * (vec3(1.0) - f0);
    diffuseColor *= 1.0 - metallic;

    float alphaRoughness = perceptualRoughness * perceptualRoughness;

    vec3 specularColor = mix(f0, baseColor.rgb, metallic);

    // Compute reflectance.
    float reflectance = max(max(specularColor.r, specularColor.g), specularColor.b);
    float reflectance90 = clamp(reflectance * 25.0, 0.0, 1.0);
    vec3 specularEnvironmentR0 = specularColor.rgb;
    vec3 specularEnvironmentR90 = vec3(1.0, 1.0, 1.0) * reflectance90;

        float len = length(lightPosition - position.xyz);
	attenuation K = getK(len);
	float lightDrop = K.C+K.L*len+K.Q*len*len;

	lightDrop *= lightDropFactor;

	float ShadowFactor = shadowFactor();
	if(ShadowFactor>pc.minAmbientFactor){
	    float lightPower = ShadowFactor*lightPowerFactor;

	    vector.eyeDirection	    = normalize(eyePosition.xyz - position.xyz);
	    vector.lightDirection   = normalize(lightPosition - position.xyz);
	    vector.normal	    = normal.xyz;
	    vector.reflect	    = -normalize(reflect(vector.eyeDirection, vector.normal));
	    vector.H		    = normalize(vector.eyeDirection + vector.lightDirection);

	    vec3  F = specularReflection(specularEnvironmentR0, specularEnvironmentR90, clamp(dot(vector.eyeDirection, vector.H), 0.0, 1.0));
	    float G = geometricOcclusion(clamp(dot(vector.normal, vector.lightDirection), 0.001, 1.0), clamp(abs(dot(vector.normal, vector.eyeDirection)), 0.001, 1.0), alphaRoughness);
	    float D = microfacetDistribution(clamp(dot(vector.normal, vector.H), 0.0, 1.0), alphaRoughness);

	    // Calculation of analytical lighting contribution
	    vec3 diffuseContrib = (1.0f - F) * diffuse(diffuseColor);
	    vec3 specContrib = F * G * D / (4.0 * clamp(dot(vector.normal, vector.lightDirection), 0.001, 1.0) * clamp(abs(dot(vector.normal, vector.eyeDirection)), 0.001, 1.0));
	    // Obtain final intensity as reflectance (BRDF) scaled by the energy of the light (cosine law)
	    vec3 color = clamp(dot(vector.normal, vector.lightDirection), 0.001, 1.0) * vec3(max(lightColor.x,textureLightColor.x),max(lightColor.y,textureLightColor.y),max(lightColor.z,textureLightColor.z)) * (diffuseContrib + specContrib);

	    const float u_OcclusionStrength = 1.0f;
	    float ao = emissiveTexture.a;
	    color = mix(color, color * ao, u_OcclusionStrength);

	    color = lightPower*color/lightDrop;
	    outColor = vec4(color, baseColor.a);
	}

	return outColor;
}


//===================================================main====================================================================//

void main()
{
    position = subpassLoad(inPositionTexture);
    normal = subpassLoad(inNormalTexture);
    baseColorTexture = subpassLoad(inBaseColorTexture);
    emissiveTexture = subpassLoad(inEmissiveTexture);

    outColor = vec4(0.0f,0.0f,0.0f,0.0f);
    outBlur = vec4(0.0f,0.0f,0.0f,0.0f);
    outBloom = vec4(0.0f,0.0f,0.0f,0.0f);

    type = lightProp.x;
    lightPowerFactor = lightProp.y;
    lightDropFactor = lightProp.z;

    if(normal.x==0.0f&&normal.y==0.0f&&normal.z==0.0f)	outColor = SRGBtoLINEAR(baseColorTexture);
    else						outColor = PBR(outColor);

        outColor += SRGBtoLINEAR(emissiveTexture);
	outBloom += SRGBtoLINEAR(emissiveTexture);

    if(outColor.x>0.95f&&outColor.y>0.95f&&outColor.y>0.95f)	outBloom += outColor;
    else							outBloom += vec4(0.0f,0.0f,0.0f,0.0f);
}

//===========================================================================================================================//
//===========================================================================================================================//

bool outsideSpotCondition(vec3 lightSpaceNDC, float type)
{
    if(type==0.0f)
	return sqrt(lightSpaceNDC.x*lightSpaceNDC.x + lightSpaceNDC.y*lightSpaceNDC.y) >= lightSpaceNDC.z;
    else
	return abs(lightSpaceNDC.x) > 1.0f || abs(lightSpaceNDC.y) > 1.0f || abs(lightSpaceNDC.z) > 1.0f;
}

float shadowFactor()
{
    float factor = pc.minAmbientFactor;

    vec3 lightSpaceNDC = fragLightPosition.xyz;
    lightSpaceNDC /= fragLightPosition.w;

    if(!outsideSpotCondition(lightSpaceNDC,type))
    {
	int n = 8; int maxNoise = 1;
	float dang = 2.0f*pi/n; float dnoise = 0.001f;
	float shadowSample = 0.0f;

	vec2 shadowMapCoord = lightSpaceNDC.xy * 0.5f + 0.5f;
	for(int j=0;j<n;j++)
	{
	    for(float noise = dnoise; noise<dnoise*(maxNoise+1); noise+=dnoise)
	    {
		vec2 dx = vec2(noise*cos(j*dang), noise*sin(j*dang));
		shadowSample += lightSpaceNDC.z-texture(shadowMap, shadowMapCoord.xy + dx).x > 0.001f ? 1.0f : 0.0f;
	    }
	}
	shadowSample /= maxNoise*n;

	textureLightColor += texture(lightTexture, shadowMapCoord.xy);

	if(1.0f - shadowSample>pc.minAmbientFactor)
	    factor = 1.0f - shadowSample;
    }

    return factor;
}

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

attenuation getK(float distance)
{
    attenuation res;
    res.C = 1.0f;
    res.L = 0.0866f*exp(-0.00144f*distance);
    res.Q = 0.0283f*exp(-0.00289f*distance);
    return res;
}
