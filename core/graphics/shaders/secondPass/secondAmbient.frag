#version 450
#define MANUAL_SRGB 1
#define MAX_LIGHT_SOURCES 20

const float pi = 3.141592653589793f, minAmbientFactor = 0.05f;


layout(location = 0)	in vec4 eyePosition;
layout(location = 1)	in vec2 fragTexCoord;
layout(location = 2)	in vec4 glPosition;

layout(input_attachment_index = 0, binding = 0) uniform subpassInput inPositionTexture;
layout(input_attachment_index = 1, binding = 1) uniform subpassInput inNormalTexture;
layout(input_attachment_index = 2, binding = 2) uniform subpassInput inBaseColorTexture;
layout(input_attachment_index = 3, binding = 3) uniform subpassInput inEmissiveTexture;

layout(set = 0, binding = 5) uniform sampler2D shadowMap[MAX_LIGHT_SOURCES];
layout(set = 0, binding = 7) uniform sampler2D lightTexture[MAX_LIGHT_SOURCES];

layout (push_constant) uniform LightPushConst
{
    int number;
} lightPC;

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

vec4		SRGBtoLINEAR(vec4 srgbIn);

void outImage1()
{
    //=========== PBR ===========//
    float metallic = subpassLoad(inNormalTexture).a;
    vec3 diffuseColor;
    vec4 baseColor = vec4(baseColorTexture.xyz,1.0f);
    baseColor = SRGBtoLINEAR(baseColor);

    vec3 f0 = vec3(0.04);

    diffuseColor = baseColor.rgb * (vec3(1.0) - f0);
    diffuseColor *= 1.0 - metallic;
    diffuseColor *= minAmbientFactor;

    outColor += vec4(diffuseColor.xyz,1.0f);
}

void shadingType0()
{
    if(normal.x==0.0f&&normal.y==0.0f&&normal.z==0.0f)	outColor = SRGBtoLINEAR(baseColorTexture);
    else						outImage1();

        outColor += SRGBtoLINEAR(emissiveTexture);
	outBloom += SRGBtoLINEAR(emissiveTexture);

    if(outColor.x>0.95f||outColor.y>0.95f||outColor.y>0.95f)	outBloom += outColor;
    else							outBloom += vec4(0.0f,0.0f,0.0f,1.0f);

}

//===================================================main====================================================================//

void main()
{
    position = subpassLoad(inPositionTexture);
    normal = subpassLoad(inNormalTexture);
    baseColorTexture = subpassLoad(inBaseColorTexture);
    emissiveTexture = subpassLoad(inEmissiveTexture);
    float depth = subpassLoad(inPositionTexture).a;

    outColor = vec4(0.0f,0.0f,0.0f,1.0f);
    outBlur = vec4(0.0f,0.0f,0.0f,1.0f);
    outBloom = vec4(0.0f,0.0f,0.0f,1.0f);

    shadingType0();
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
