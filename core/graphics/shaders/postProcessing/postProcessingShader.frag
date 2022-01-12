#version 450

layout(set = 0, binding = 0) uniform sampler2D Sampler;
layout(set = 0, binding = 1) uniform sampler2D bloomSampler;
layout(set = 0, binding = 2) uniform sampler2D godRaysSampler;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 bloomOutColor;
layout(location = 2) out vec4 godRaysOutColor;

const float pi = 3.141592653589793f;

vec4 blur(sampler2D bloomSampler, vec2 TexCoord)
{
    vec2 textel = 1.0 / textureSize(bloomSampler, 0);
    vec3 Color = vec3(0.0f);
    int h = 5;
    float I = 1.0f/(2.0f*h+1.0f)/(2.0f*h+1.0f);
    for(int i=-h;i<h+1;i++)
    {
	for(int j=-h;j<h+1;j++)
	{
	    Color += texture(bloomSampler, TexCoord + vec2(i*textel.x,j*textel.y) ).xyz * I;
	}
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
    float factor = exp( - 2.0f * (NDCfragCoord.x*NDCfragCoord.x + NDCfragCoord.y*NDCfragCoord.y));

    outColor = texture(Sampler,fragTexCoord);
    bloomOutColor = blur(bloomSampler,fragTexCoord);
    godRaysOutColor = godRays(godRaysSampler,fragTexCoord);
}
