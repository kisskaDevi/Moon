#version 450
#define MANUAL_SRGB 1
#define MAX_LIGHT_SOURCES 8

const float pi = 3.141592653589793f;
float minAmbientFactor = 0.05f;

layout(set = 0, binding = 2) uniform sampler2D shadowMap[MAX_LIGHT_SOURCES];

layout(set = 3, binding = 0) uniform sampler2D baseColorTexture;
layout(set = 3, binding = 1) uniform sampler2D metallicRoughnessTexture;
layout(set = 3, binding = 2) uniform sampler2D normalTexture;
layout(set = 3, binding = 3) uniform sampler2D occlusionTexture;
layout(set = 3, binding = 4) uniform sampler2D emissiveTexture;

layout(location = 0)	in vec3 position;
layout(location = 1)	in vec3 normal;
layout(location = 2)	in vec2 UV0;
layout(location = 3)	in vec2 UV1;
layout(location = 4)	in vec4 eyePosition;
layout(location = 5)	in mat3 TBN;

layout(location = 8)	in vec4 lightPosition[MAX_LIGHT_SOURCES];
layout(location = 16)	in vec4 fragLightPosition[MAX_LIGHT_SOURCES];
layout(location = 24)	in vec4 lightColor[MAX_LIGHT_SOURCES];

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outFilterColor;
layout(location = 2) out vec4 outGodRays;

layout (push_constant) uniform Material
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
} material;

struct Vector{
    vec3 eyeDirection;
    vec3 lightDirection;
    vec3 normal;
    vec3 reflect;
    vec3 H;
}vector;

//===================================================functions====================================================================//

vec3 getNormal();

float shadowFactor(int i);

vec4 SRGBtoLINEAR(vec4 srgbIn);

float DistributionGGX(vec3 N, vec3 H, float roughness);
float GeometrySchlickGGX(float NdotV, float roughness);
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness);
vec3 fresnelSchlick(float cosTheta, vec3 F0);


//===================================================outImages====================================================================//

void outImage1()
{
    outColor = vec4(0.0f,0.0f,0.0f,1.0f);

    float metallic  = texture(metallicRoughnessTexture, UV0).b;
    float roughness = texture(metallicRoughnessTexture, UV0).g;
    float ao	    = texture(occlusionTexture, UV0).r;

    float specularFactor = 16.0f;
    float lightDropFactor = 0.01f;

    vec4 diffMatColor = SRGBtoLINEAR(texture(baseColorTexture, UV0));

    vec3 F0 = vec3(0.04f);
    F0 = mix(F0, diffMatColor.xyz, metallic);

    int	lightCount = 0;
    for(int i=0;i<MAX_LIGHT_SOURCES;i++)
    {
	float lightPower = 1.0f*shadowFactor(i);
	if(lightPower>minAmbientFactor)
	{
	    float len = length(lightPosition[i].xyz - position);
	    float lightDrop = 1.0f+lightDropFactor*pow(len,2.0f);

	    vector.eyeDirection	    = normalize(eyePosition.xyz - position);
	    vector.lightDirection   = normalize(lightPosition[i].xyz - position);
	    vector.normal	    = material.normalTextureSet > -1 ? getNormal() : normalize(normal);
	    vector.reflect	    = reflect(-vector.lightDirection, vector.normal);
	    vector.H		    = normalize(vector.eyeDirection + vector.lightDirection);

	        float NDF   = DistributionGGX(vector.normal, vector.H, roughness);
		float G	    = GeometrySmith(vector.normal, vector.eyeDirection, vector.lightDirection, roughness);
		vec3 F	    = fresnelSchlick(max(dot(vector.H, vector.eyeDirection), 0.0), F0);

		vec3 kS = F;
		vec3 kD = vec3(1.0f) - kS;
		kD *= 1.0f - metallic;

		vec3 numerator    = NDF * G * F;
		float denominator = 4.0f * max(dot(vector.normal, vector.eyeDirection), 0.0f) * max(dot(vector.normal, vector.lightDirection), 0.0f) + 0.0001f;
		vec3 specular     = numerator / denominator;

	    vec4 diffuseColor = vec4(kD * diffMatColor.xyz,1.0f);
	    vec4 specularColor = vec4(specular * pow(max(0.0f,dot(vector.reflect, vector.eyeDirection)),specularFactor),1.0f);

	    vec4 resultColor = vec4(0.0f,0.0f,0.0f,1.0f);
	    resultColor += vec4(diffuseColor.xyz/lightDrop,1.0f);
	    resultColor += vec4(specularColor.xyz/lightDrop,1.0f);

	    lightCount++;
	    vec4 resultLightColor = vec4(lightPower*lightColor[i].xyz,0.0f);

	    resultColor *= resultLightColor;

	    outColor = vec4(vec3(max(outColor,resultColor).xyz),1.0f);
	}
    }
    if(lightCount==0)
    {
	vec4 ambientColor = minAmbientFactor*diffMatColor;
	outColor = ambientColor;
    }
}

void outImage2()
{
    outFilterColor = SRGBtoLINEAR(texture(emissiveTexture, UV0));
}

void outImage3()
{
    outGodRays = SRGBtoLINEAR(texture(emissiveTexture, UV0));
}

//===================================================main====================================================================//

void main()
{
    outImage1();
    outImage2();
    outImage3();
}

vec3 getNormal()
{
    vec3 tangentNormal = normalize(texture(normalTexture, material.normalTextureSet == 0 ? UV0 : UV1).xyz * 2.0f - 1.0f);

    return normalize(TBN * tangentNormal);
}

float shadowFactor(int i)
{
    vec3 lightSpaceNDC = fragLightPosition[i].xyz;
    lightSpaceNDC /= fragLightPosition[i].w;

    if(lightColor[i].w==0.0f)
    {
	if( lightSpaceNDC.x*lightSpaceNDC.x +
	    lightSpaceNDC.y*lightSpaceNDC.y +
	    lightSpaceNDC.z*lightSpaceNDC.z > 2.0f )
	{return minAmbientFactor;}
    }else
    {
	if( abs(lightSpaceNDC.x) > 1.0f ||
	    abs(lightSpaceNDC.y) > 1.0f ||
	    abs(lightSpaceNDC.z) > 1.0f )
	{return minAmbientFactor;}
    }

    vec2 shadowMapCoord = lightSpaceNDC.xy * 0.5f + 0.5f;

        int n = 8; int maxNoise = 1;
	float dang = 2.0f*pi/n; float dnoise = 0.001f;
	float shadowSample = 0.0f;
	for(int j=0;j<n;j++)
	{
	    for(float noise = dnoise; noise<dnoise*(maxNoise+1); noise+=dnoise)
	    {
		vec2 dx = vec2(noise*cos(j*dang), noise*sin(j*dang));
		shadowSample += lightSpaceNDC.z-texture(shadowMap[i], shadowMapCoord.xy + dx).x > 0.0001f ? 1.0f : 0.0f;
	    }
	}
	shadowSample /= maxNoise*n;

	if(1.0f - shadowSample<minAmbientFactor)
	{return minAmbientFactor;}
	return 1.0f - shadowSample;

//    if(lightSpaceNDC.z-texture(shadowMap[i], shadowMapCoord.xy).x > 0.0001f)
//    {return minAmbientFactor;}

//    return 1.0f;

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

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0f);
    float NdotH2 = NdotH*NdotH;

    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = pi * denom * denom;

    return num / denom;
}
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0f);
    float k = (r*r) / 8.0f;

    float num   = NdotV;
    float denom = NdotV * (1.0f - k) + k;

    return num / denom;
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0f);
    float NdotL = max(dot(N, L), 0.0f);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0f - F0) * pow(clamp(1.0f - cosTheta, 0.0f, 1.0f), 5.0f);
}
