#version 450
#define MANUAL_SRGB 1
#define MAX_LIGHT_SOURCES 8

const float pi = 3.141592653589793f;
float minAmbientFactor = 0.05f;

layout(location = 0)	in vec4 eyePosition;
layout(location = 1)	in vec2 fragTexCoord;

layout(input_attachment_index = 0, binding = 0) uniform subpassInput inPositionTexture;
layout(input_attachment_index = 1, binding = 1) uniform subpassInput inBaseColorTexture;
layout(input_attachment_index = 2, binding = 2) uniform subpassInput inMetallicRoughnessTexture;
layout(input_attachment_index = 3, binding = 3) uniform subpassInput inNormalTexture;
layout(input_attachment_index = 4, binding = 4) uniform subpassInput inOcclusionTexture;
layout(input_attachment_index = 5, binding = 5) uniform subpassInput inEmissiveTexture;

layout(set = 0, binding = 6) uniform LightUniformBufferObject
{
    mat4 projView;
    vec4 position;
    vec4 lightColor;
    int type;
    int enableShadow;
} lightubo[MAX_LIGHT_SOURCES];

layout(set = 0, binding = 7) uniform sampler2D shadowMap[MAX_LIGHT_SOURCES];

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outBloom;
layout(location = 2) out vec4 outGodRays;

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
    for(int i=0;i<MAX_LIGHT_SOURCES;i++)
    {
	lightPosition[i] = lightubo[i].position.xyz;
	fragLightPosition[i] = lightubo[i].projView * vec4(position.xyz,1.0f);
	lightColor[i] = lightubo[i].lightColor;
    }

    outColor = vec4(0.0f,0.0f,0.0f,1.0f);

    float metallic  = metallicRoughnessTexture.b;
    float roughness = metallicRoughnessTexture.g;
    float ao	    = occlusionTexture.r;

    float specularFactor = 16.0f;
    float lightDropFactor = 0.01f;

    vec4 diffMatColor = SRGBtoLINEAR(baseColorTexture);

    vec3 F0 = vec3(0.04f);
    F0 = mix(F0, diffMatColor.xyz, metallic);

    int	lightCount = 0;
    for(int i=0;i<MAX_LIGHT_SOURCES;i++)
    {
	float lightPower = 1.0f*shadowFactor(i);
	if(lightPower>minAmbientFactor)
	{
	    float len = length(lightPosition[i] - position.xyz);
	    float lightDrop = 1.0f+lightDropFactor*pow(len,2.0f);

	    vector.eyeDirection	    = normalize(eyePosition.xyz - position.xyz);
	    vector.lightDirection   = normalize(lightPosition[i] - position.xyz);
	    vector.normal	    = normal;
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

void shadingType0()
{
    outImage1();
    outBloom = SRGBtoLINEAR(emissiveTexture);
}

void shadingType1()
{
    outColor = SRGBtoLINEAR(baseColorTexture);
    outBloom = outColor;
}

void shadingType2()
{
    outColor = SRGBtoLINEAR(baseColorTexture);
    outGodRays = SRGBtoLINEAR(emissiveTexture);
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
    normal = subpassLoad(inNormalTexture).rgb;
    baseColorTexture = subpassLoad(inBaseColorTexture);
    metallicRoughnessTexture = subpassLoad(inMetallicRoughnessTexture);
    occlusionTexture = subpassLoad(inOcclusionTexture);
    emissiveTexture = subpassLoad(inEmissiveTexture);

    int type = int(position.a);

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
}

//===========================================================================================================================//
//===========================================================================================================================//
float shadowFactor(int i)
{
    vec3 lightSpaceNDC = fragLightPosition[i].xyz;
    lightSpaceNDC /= fragLightPosition[i].w;

    if(lightubo[i].type==0.0f)
    {
	if( lightSpaceNDC.x*lightSpaceNDC.x +
	    lightSpaceNDC.y*lightSpaceNDC.y > 1.0f )
	{return minAmbientFactor;}
    }else
    {
	if( abs(lightSpaceNDC.x) > 1.0f ||
	    abs(lightSpaceNDC.y) > 1.0f )
	{return minAmbientFactor;}
    }

    if(lightubo[i].enableShadow==1.0)
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
		shadowSample += lightSpaceNDC.z-texture(shadowMap[i], shadowMapCoord.xy + dx).x > 0.0001f ? 1.0f : 0.0f;
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
