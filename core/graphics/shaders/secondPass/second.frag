#version 450
#define MANUAL_SRGB 1
#define MAX_LIGHT_SOURCES 10
#define MAX_NODE_COUNT 256

const float pi = 3.141592653589793f, minAmbientFactor = 0.05f;
const float PBR_WORKFLOW_METALLIC_ROUGHNESS = 0.0;
const float PBR_WORKFLOW_SPECULAR_GLOSINESS = 1.0f;
const float c_MinRoughness = 0.04;
const float lightDropFactor = 0.001f;
const float lightPowerFactor = 2.0f;

int type, number;

layout(location = 0)	in vec4 eyePosition;
layout(location = 1)	in vec2 fragTexCoord;

layout(input_attachment_index = 0, binding = 0) uniform subpassInput inPositionTexture;
layout(input_attachment_index = 1, binding = 1) uniform subpassInput inNormalTexture;
layout(input_attachment_index = 2, binding = 2) uniform subpassInput inBaseColorTexture;
layout(input_attachment_index = 3, binding = 3) uniform subpassInput inMetallicRoughnessTexture;
layout(input_attachment_index = 4, binding = 4) uniform subpassInput inOcclusionTexture;
layout(input_attachment_index = 5, binding = 5) uniform subpassInput inEmissiveTexture;

struct LightBufferObject
{
    mat4 projView;
    vec4 position;
    vec4 lightColor;
    int type;
    int enableShadow;
};

struct Material
{
    vec4 baseColorFactor;
    vec4 emissiveFactor;
    vec4 diffuseFactor;
    vec4 specularFactor;
    float workflow;
    int baseColorTextureSet;
    int physicalDescriptorTextureSet;
    int normalTextureSet;
    int occlusionTextureSet;
    int emissiveTextureSet;
    float metallicFactor;
    float roughnessFactor;
    float alphaMask;
    float alphaMaskCutoff;
    int number;
};

layout(set = 0, binding = 6) uniform LightUniformBufferObject
{
    LightBufferObject ubo[MAX_LIGHT_SOURCES];
} light;

layout(set = 0, binding = 7) uniform sampler2D shadowMap[MAX_LIGHT_SOURCES];

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outBloom;

layout(set = 0, binding = 9) uniform MaterialUniformBufferObject
{
    Material ubo[MAX_NODE_COUNT];
} material;

struct Vector{
    vec3 eyeDirection;
    vec3 lightDirection;
    vec3 normal;
    vec3 reflect;
    vec3 H;
}vector;

vec4 position;
vec3 normal;
vec4 baseColorTexture;
vec4 metallicRoughnessTexture;
vec4 occlusionTexture;
vec4 emissiveTexture;

vec3 lightPosition[MAX_LIGHT_SOURCES];
vec4 fragLightPosition[MAX_LIGHT_SOURCES];
vec4 lightColor[MAX_LIGHT_SOURCES];


//===================================================functions====================================================================//

float shadowFactor(int i);

vec4 SRGBtoLINEAR(vec4 srgbIn);
vec3 fresnelSchlick(float cosTheta, vec3 F0);

//===================================================outImages====================================================================//

float convertMetallic(vec3 diffuse, vec3 specular, float maxSpecular) {
        float perceivedDiffuse = sqrt(0.299 * diffuse.r * diffuse.r + 0.587 * diffuse.g * diffuse.g + 0.114 * diffuse.b * diffuse.b);
	float perceivedSpecular = sqrt(0.299 * specular.r * specular.r + 0.587 * specular.g * specular.g + 0.114 * specular.b * specular.b);
	if (perceivedSpecular < c_MinRoughness) {
	        return 0.0;
	}
	float a = c_MinRoughness;
	float b = perceivedDiffuse * (1.0 - maxSpecular) / (1.0 - c_MinRoughness) + perceivedSpecular - 2.0 * c_MinRoughness;
	float c = c_MinRoughness - perceivedSpecular;
	float D = max(b * b - 4.0 * a * c, 0.0);
	return clamp((-b + sqrt(D)) / (2.0 * a), 0.0, 1.0);
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

void outImage1()
{
    for(int i=0;i<MAX_LIGHT_SOURCES;i++)
    {
	lightPosition[i] = light.ubo[i].position.xyz;
	fragLightPosition[i] = light.ubo[i].projView * vec4(position.xyz,1.0f);
	lightColor[i] = light.ubo[i].lightColor;
    }

    outColor = vec4(0.0f,0.0f,0.0f,1.0f);

    float perceptualRoughness;
    float metallic;
    vec3 diffuseColor;
    vec4 baseColor;

    vec3 f0 = vec3(0.04);

    if (material.ubo[number].workflow == PBR_WORKFLOW_METALLIC_ROUGHNESS) {
	    // Metallic and Roughness material properties are packed together
	    // In glTF, these factors can be specified by fixed scalar values
	    // or from a metallic-roughness map
	    perceptualRoughness = material.ubo[number].roughnessFactor;
	    metallic		= material.ubo[number].metallicFactor;
	    if (material.ubo[number].physicalDescriptorTextureSet > -1) {
		    // Roughness is stored in the 'g' channel, metallic is stored in the 'b' channel.
		    // This layout intentionally reserves the 'r' channel for (optional) occlusion map data
		    vec4 mrSample = metallicRoughnessTexture;
		    perceptualRoughness = mrSample.g * perceptualRoughness;
		    metallic = mrSample.b * metallic;
	    } else {
		    perceptualRoughness = clamp(perceptualRoughness, c_MinRoughness, 1.0);
		    metallic = clamp(metallic, 0.0, 1.0);
	    }
	    // Roughness is authored as perceptual roughness; as is convention,
	    // convert to material roughness by squaring the perceptual roughness [2].

	    // The albedo may be defined from a base texture or a flat color
	    if (material.ubo[number].baseColorTextureSet > -1) {
		    baseColor = SRGBtoLINEAR(baseColorTexture) * material.ubo[number].baseColorFactor;
	    } else {
		    baseColor = material.ubo[number].baseColorFactor;
	    }
    }

    if (material.ubo[number].workflow == PBR_WORKFLOW_SPECULAR_GLOSINESS) {
	    // Values from specular glossiness workflow are converted to metallic roughness
	    if (material.ubo[number].physicalDescriptorTextureSet > -1) {
		    perceptualRoughness = 1.0 - metallicRoughnessTexture.a;
	    } else {
		    perceptualRoughness = 0.0;
	    }

	    const float epsilon = 1e-6;

	    vec4 diffuse = SRGBtoLINEAR(baseColorTexture);
	    vec3 specular = SRGBtoLINEAR(metallicRoughnessTexture).rgb;

	    float maxSpecular = max(max(specular.r, specular.g), specular.b);

	    // Convert metallic value from specular glossiness inputs
	    metallic = convertMetallic(diffuse.rgb, specular, maxSpecular);

	    vec3 baseColorDiffusePart = diffuse.rgb * ((1.0 - maxSpecular) / (1 - c_MinRoughness) / max(1 - metallic, epsilon)) * material.ubo[number].diffuseFactor.rgb;
	    vec3 baseColorSpecularPart = specular - (vec3(c_MinRoughness) * (1 - metallic) * (1 / max(metallic, epsilon))) * material.ubo[number].specularFactor.rgb;
	    baseColor = vec4(mix(baseColorDiffusePart, baseColorSpecularPart, metallic * metallic), diffuse.a);

    }

    diffuseColor = baseColor.rgb * (vec3(1.0) - f0);
    diffuseColor *= 1.0 - metallic;

    float alphaRoughness = perceptualRoughness * perceptualRoughness;

    vec3 specularColor = mix(f0, baseColor.rgb, metallic);

    // Compute reflectance.
    float reflectance = max(max(specularColor.r, specularColor.g), specularColor.b);
    float reflectance90 = clamp(reflectance * 25.0, 0.0, 1.0);
    vec3 specularEnvironmentR0 = specularColor.rgb;
    vec3 specularEnvironmentR90 = vec3(1.0, 1.0, 1.0) * reflectance90;

    int	lightCount = 0;
    for(int i=0;i<MAX_LIGHT_SOURCES;i++)
    {
	float lightPower = shadowFactor(i);
	if(lightPower>minAmbientFactor)
	{
	    float lightPower = lightPowerFactor * lightPower;
	    float len = length(lightPosition[i] - position.xyz);
	    float lightDrop = 1.0f+lightDropFactor*pow(len,2.0f);

	    vector.eyeDirection	    = normalize(eyePosition.xyz - position.xyz);
	    vector.lightDirection   = normalize(lightPosition[i] - position.xyz);
	    vector.normal	    = normal;
	    vector.reflect	    = reflect(-vector.lightDirection, vector.normal);
	    vector.H		    = normalize(vector.eyeDirection + vector.lightDirection);

	    vec3 F = specularReflection(specularEnvironmentR0, specularEnvironmentR90, clamp(dot(vector.eyeDirection, vector.H), 0.0, 1.0));
	    float G = geometricOcclusion(clamp(dot(vector.normal, vector.lightDirection), 0.001, 1.0), clamp(abs(dot(vector.normal, vector.eyeDirection)), 0.001, 1.0), alphaRoughness);
	    float D = microfacetDistribution(clamp(dot(vector.normal, vector.H), 0.0, 1.0), alphaRoughness);

	    // Calculation of analytical lighting contribution
	    vec3 diffuseContrib = (1.0 - F) * diffuse(diffuseColor);
	    vec3 specContrib = F * G * D / (4.0 * clamp(dot(vector.normal, vector.lightDirection), 0.001, 1.0) * clamp(abs(dot(vector.normal, vector.eyeDirection)), 0.001, 1.0));
	    // Obtain final intensity as reflectance (BRDF) scaled by the energy of the light (cosine law)
	    vec3 color = clamp(dot(vector.normal, vector.lightDirection), 0.001, 1.0) * light.ubo[i].lightColor.rgb * (diffuseContrib + specContrib);

	    const float u_OcclusionStrength = 1.0f;
	    // Apply optional PBR terms for additional (optional) shading
	    if (material.ubo[number].occlusionTextureSet > -1) {
		    float ao = occlusionTexture.r;
		    color = mix(color, color * ao, u_OcclusionStrength);
	    }

	    color = lightPower*color/lightDrop;
	    outColor = vec4(max(color.r,outColor.r),max(color.g,outColor.g),max(color.b,outColor.b), baseColor.a);
	    lightCount++;
	}
    }
    const float u_EmissiveFactor = 1.0f;
    if(lightCount==0)
    {
	if (material.ubo[number].emissiveTextureSet > -1) {
	    vec4 emissive = vec4(SRGBtoLINEAR(emissiveTexture).rgb * u_EmissiveFactor,0.0f);
	    outColor += emissive;
	    outColor = vec4(outColor.rgb, baseColor.a);
	}
	else{
	    outColor = minAmbientFactor*vec4(diffuseColor, baseColor.a);
	}
    }
    else
    {
	if (material.ubo[number].emissiveTextureSet > -1) {
	    vec4 emissive = vec4(SRGBtoLINEAR(emissiveTexture).rgb * u_EmissiveFactor,0.0f);
	    outColor += emissive;
	}
    }
}

void shadingType0()
{
    outImage1();
    outBloom = SRGBtoLINEAR(emissiveTexture);
    const float u_EmissiveFactor = 1.0f;
    if (material.ubo[number].emissiveTextureSet > -1) {
	    vec3 emissive = SRGBtoLINEAR(emissiveTexture).rgb * u_EmissiveFactor;
	    outBloom += vec4(emissive,1.0f);
    }
}

void shadingType1()
{
    outColor = SRGBtoLINEAR(baseColorTexture);
    outBloom = outColor;
}

void shadingType2()
{
    outColor = SRGBtoLINEAR(baseColorTexture);
}

void shadingType3()
{
    outColor = baseColorTexture;
    outBloom = vec4(0.0f);
}

//===================================================main====================================================================//

void main()
{
    position = subpassLoad(inPositionTexture);
    normal = subpassLoad(inNormalTexture).xyz;
    baseColorTexture = subpassLoad(inBaseColorTexture);
    metallicRoughnessTexture = subpassLoad(inMetallicRoughnessTexture);
    occlusionTexture = subpassLoad(inOcclusionTexture);
    emissiveTexture = subpassLoad(inEmissiveTexture);
    type = int(position.a);
    number = int(subpassLoad(inNormalTexture).a);


    switch(type)
    {
        case 0:
	   shadingType0();
	   break;
        case 1:
	   shadingType1();
	   break;
        case 2:
	   shadingType2();
	   break;
        case 3:
	   shadingType3();
	   break;
    }
    //outColor = vec4(material.ubo[int(subpassLoad(inNormalTexture).a)].number/128.0f,0.0f,0.0f,1.0f);
    //outColor = vec4(normal,1.0f);
}

//===========================================================================================================================//
//===========================================================================================================================//

float shadowFactor(int i)
{
    vec3 lightSpaceNDC = fragLightPosition[i].xyz;
    lightSpaceNDC /= fragLightPosition[i].w;

    if(light.ubo[i].type==0.0f)
    {
	    if( lightSpaceNDC.x*lightSpaceNDC.x +
	        lightSpaceNDC.y*lightSpaceNDC.y -
	        lightSpaceNDC.z*lightSpaceNDC.z > 0.0f)
	    {return minAmbientFactor;}
    }else
    {
	if( abs(lightSpaceNDC.x) > 1.0f ||
	    abs(lightSpaceNDC.y) > 1.0f ||
	    abs(lightSpaceNDC.z) > 1.0f)
	{return minAmbientFactor;}
    }

    if(light.ubo[i].enableShadow==1.0)
    {
	vec2 shadowMapCoord = lightSpaceNDC.xy * 0.5f + 0.5f;

	int n = 8; int maxNoise = 1;
	float dang = 2.0f*pi/n; float dnoise = 0.001f;
	float shadowSample = 0.0f;
	for(int j=0;j<n;j++)
	{
	    for(float noise = dnoise; noise<dnoise*(maxNoise+1); noise+=dnoise)
	    {
		vec2 dx = vec2(noise*cos(j*dang), noise*sin(j*dang));
		shadowSample += lightSpaceNDC.z-texture(shadowMap[i], shadowMapCoord.xy + dx).x > 0.001f ? 1.0f : 0.0f;
	    }
	}
	shadowSample /= maxNoise*n;

	if(1.0f - shadowSample<minAmbientFactor)
	{return minAmbientFactor;}
	return 1.0f - shadowSample;

//	    if(lightSpaceNDC.z>texture(shadowMap[i], shadowMapCoord.xy).x )
//	    {return minAmbientFactor;}
    }
    return 1.0f;
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
	return vec4(linOut,srgbIn.w);;
        #else //MANUAL_SRGB
        return srgbIn;
        #endif //MANUAL_SRGB
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0f - F0) * pow(clamp(1.0f - cosTheta, 0.0f, 1.0f), 5.0f);
}
