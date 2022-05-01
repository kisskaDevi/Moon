#version 450

layout(set = 0, binding = 0) uniform sampler2D Sampler;
layout(set = 0, binding = 1) uniform sampler2D bloomSampler;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

const float pi = 3.141592653589793f;

vec4 blur(sampler2D Sampler, vec2 TexCoord)
{
    float sigma = 2.0 * textureSize(Sampler, 0).y;
    vec2 textel = 1.0 / textureSize(Sampler, 0);
    vec3 Color = texture(bloomSampler, TexCoord).xyz /sqrt(pi*sigma);
    int h = 20;
    float Norm = 1.0f/sqrt(pi*sigma);
    for(int i=-h;i<h+1;i+=2)
    {
	float I1 = Norm * exp( -(i*textel.y*i*textel.y)/sigma);
	float I2 = Norm * exp( -((i+1)*textel.y*(i+1)*textel.y)/sigma);
	float y = (I1*i+I2*(i+1))*textel.y/(I1+I2);
	float I = Norm * exp( -(y*y)/sigma);
	Color += texture(bloomSampler, TexCoord + vec2(0.0f,y) ).xyz * I;
	Color += texture(bloomSampler, TexCoord - vec2(0.0f,y) ).xyz * I;
    }
    return vec4(Color, 1.0);
}

vec4 godRays(sampler2D bloomSampler, vec2 TexCoord)
{
    int Samples = 128;
    float Intensity = 0.125f, Decay = 0.96875f;
    vec2 Direction = vec2(0.5f) - TexCoord;
    Direction /= Samples;
    vec3 Color = texture(bloomSampler, TexCoord).xyz;

    for(int Sample = 0; Sample < Samples; Sample++)
    {
	Color += texture(bloomSampler, TexCoord).xyz * Intensity;
	Intensity *= Decay;
	TexCoord += Direction;
    }

    return vec4(Color, 1.0);
}

void main()
{
    vec2 NDCfragCoord = fragTexCoord * 2.0f - 1.0f;
    float factor = exp( - 1.0f * (NDCfragCoord.x*NDCfragCoord.x + NDCfragCoord.y*NDCfragCoord.y));

    outColor = factor*vec4(0.6f,0.6f,0.8f,1.0f)*texture(Sampler,fragTexCoord);
    outColor += blur(bloomSampler,fragTexCoord);
}
