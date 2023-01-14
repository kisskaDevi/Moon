#version 450
#define bloomCount 8

layout(set = 0, binding = 0) uniform sampler2D Sampler;
layout(set = 0, binding = 1) uniform sampler2D blurSampler;
layout(set = 0, binding = 2) uniform sampler2D bloomSampler[bloomCount];
layout(set = 0, binding = 3) uniform sampler2D sslrSampler;
layout(set = 0, binding = 4) uniform sampler2D ssaoSampler;

vec4 colorBloomFactor[bloomCount] = vec4[](
    vec4(1.0f,0.0f,0.0f,1.0f),
    vec4(0.0f,0.0f,1.0f,1.0f),
    vec4(0.0f,1.0f,0.0f,1.0f),
    vec4(1.0f,0.0f,0.0f,1.0f),
    vec4(0.0f,0.0f,1.0f,1.0f),
    vec4(0.0f,1.0f,0.0f,1.0f),
    vec4(1.0f,0.0f,0.0f,1.0f),
    vec4(1.0f,1.0f,1.0f,1.0f)
);

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

layout (push_constant) uniform PC
{
    float blitFactor;
}pc;

const float pi = 3.141592653589793f;

vec4 radialBlur(sampler2D Sampler, vec2 TexCoord)
{
    int Samples = 128;
    float Intensity = 0.125f, Decay = 0.96875f;
    vec2 Direction = vec2(0.5f) - TexCoord;
    Direction /= Samples;
    vec3 Color = texture(Sampler, TexCoord).xyz;

    for(int Sample = 0; Sample < Samples; Sample++)
    {
	Color += texture(Sampler, TexCoord).xyz * Intensity;
	Intensity *= Decay;
	TexCoord += Direction;
    }

    return vec4(Color, 1.0);
}

vec4 bloom()
{
    float blitFactor = pc.blitFactor;
    vec4 bloomColor = vec4(0.0f);
    float invBlitFactor = 1.0f/blitFactor;
    for(int i=0;i<bloomCount;i++){
	vec2 coord = fragTexCoord*invBlitFactor;
	bloomColor += colorBloomFactor[i]*texture(bloomSampler[i],coord)*exp(0.01*i*i);
	invBlitFactor/=blitFactor;
    }

    return bloomColor;
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

void main()
{
    outColor = vec4(0.0f,0.0f,0.0f,0.0f);

    outColor += texture(Sampler,fragTexCoord);
    outColor += texture(blurSampler,fragTexCoord);
    outColor += bloom();
    outColor += vec4(texture(sslrSampler,fragTexCoord).xyz,0.0f);
    //outColor += vec4(texture(ssaoSampler,fragTexCoord).xyz,0.0f);
}
