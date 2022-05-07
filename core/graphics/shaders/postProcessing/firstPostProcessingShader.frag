#version 450

layout(set = 0, binding = 0) uniform sampler2D bloomSampler;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 bloomColor;

const float pi = 3.141592653589793f;

vec4 blur(sampler2D bloomSampler, vec2 TexCoord)
{
    float sigma = 2.0 * textureSize(bloomSampler, 0).x;
    vec2 textel = 1.0 / textureSize(bloomSampler, 0);
    vec3 Color = texture(bloomSampler, TexCoord).xyz /sqrt(pi*sigma);
    int h = 20;
    float Norm = 1.0f/sqrt(pi*sigma);
    for(int i=1;i<h+1;i+=2)
    {
	float I1 = Norm * exp( -(i*textel.x*i*textel.x)/sigma);
	float I2 = Norm * exp( -((i+1)*textel.x*(i+1)*textel.x)/sigma);
	float x = (I1*i+I2*(i+1))*textel.x/(I1+I2);
	float I = Norm * exp(-(x*x)/sigma);
	Color += texture(bloomSampler, TexCoord + vec2(x,0.0f) ).xyz * I;
	Color += texture(bloomSampler, TexCoord - vec2(x,0.0f) ).xyz * I;
    }
    return vec4(Color, 1.0);
}

void main()
{
    outColor = vec4(0.2f);
    bloomColor = blur(bloomSampler,fragTexCoord);
}
