#version 450

layout(set = 0, binding = 0) uniform sampler2D blurSampler;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outBlur;

const float pi = 3.141592653589793f;

vec4 blur(sampler2D blurSampler, vec2 TexCoord)
{
    float sigma = 1.0 * textureSize(blurSampler, 0).x;
    vec2 textel = 1.0 / textureSize(blurSampler, 0);
    vec4 Color = texture(blurSampler, TexCoord) /sqrt(pi*sigma);
    int h = 10;
    float Norm = 1.0f/sqrt(pi*sigma);
    for(int i=1;i<h+1;i+=2)
    {
	float I1 = Norm * exp( -(i*textel.x*i*textel.x)/sigma);
	float I2 = Norm * exp( -((i+1)*textel.x*(i+1)*textel.x)/sigma);
	float x = (I1*i+I2*(i+1))*textel.x/(I1+I2);
	float I = Norm * exp(-(x*x)/sigma);
	Color += texture(blurSampler, TexCoord + vec2(x,0.0f) ) * I;
	Color += texture(blurSampler, TexCoord - vec2(x,0.0f) ) * I;
    }
    return Color;
}

void main()
{
    outColor = vec4(0.0f);
    outBlur = vec4(0.0f);
    outBlur += texture(blurSampler,fragTexCoord);
}
