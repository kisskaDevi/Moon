#version 450
#define MANUAL_SRGB 1

const float pi = 3.141592653589793f;
float minAmbientFactor = 0.1f;

layout(set = 0, binding = 2) uniform sampler2D shadowMap[6];

layout(set = 3, binding = 0) uniform sampler2D baseColorTexture;
layout(set = 3, binding = 1) uniform sampler2D metallicRoughnessTexture;
layout(set = 3, binding = 2) uniform sampler2D normalTexture;
layout(set = 3, binding = 3) uniform sampler2D occlusionTexture;
layout(set = 3, binding = 4) uniform sampler2D emissiveTexture;

layout(location = 0)	in vec3 position;
layout(location = 1)	in vec2 UV0;
layout(location = 2)	in vec2 UV1;

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

vec4 SRGBtoLINEAR(vec4 srgbIn);


//===================================================outImages====================================================================//

void outImage1()
{
}

void outImage2()
{
    vec4 diffMatColor = SRGBtoLINEAR(texture(baseColorTexture, UV0));

    outFilterColor = diffMatColor;
}

void outImage3()
{
}

//===================================================main====================================================================//

void main()
{
    outImage2();
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
