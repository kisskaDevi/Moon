#version 450
#define bloomCount 8

layout(set = 0, binding = 0) uniform sampler2D Sampler;
layout(set = 0, binding = 1) uniform sampler2D blurSampler;
layout(set = 0, binding = 2) uniform sampler2D sslrSampler;
layout(set = 0, binding = 3) uniform sampler2D bloomSampler[bloomCount];

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

vec4 blur(sampler2D Sampler, vec2 TexCoord)
{
    float sigma = 1.0 * textureSize(Sampler, 0).y;
    vec2 textel = 1.0 / textureSize(Sampler, 0);
    vec4 Color = texture(Sampler, TexCoord) /sqrt(pi*sigma);
    int h = 20;
    float Norm = 1.0f/sqrt(pi*sigma);
    for(int i=-h;i<h+1;i+=2)
    {
	float I1 = Norm * exp( -(i*textel.y*i*textel.y)/sigma);
	float I2 = Norm * exp( -((i+1)*textel.y*(i+1)*textel.y)/sigma);
	float y = (I1*i+I2*(i+1))*textel.y/(I1+I2);
	float I = Norm * exp( -(y*y)/sigma);
	Color += texture(Sampler, TexCoord + vec2(0.0f,y) ) * I;
	Color += texture(Sampler, TexCoord - vec2(0.0f,y) ) * I;
    }
    return Color;
}

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


void main()
{
    outColor = vec4(0.0f,0.0f,0.0f,0.0f);

    outColor += vec4(texture(Sampler,fragTexCoord).xyz,0.0f);
    outColor += blur(blurSampler,fragTexCoord);
    outColor += bloom();
    //outColor += vec4(texture(sslrSampler,fragTexCoord).xyz,0.0f);
}
