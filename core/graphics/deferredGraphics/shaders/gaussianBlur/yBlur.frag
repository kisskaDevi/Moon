#version 450

layout(set = 0, binding = 0) uniform sampler2D blurSampler;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

const float pi = 3.141592653589793f;

vec4 blur(sampler2D Sampler, vec2 TexCoord)
{
    float sigma = 1.0 * textureSize(Sampler, 0).y;
    vec2 textel = 1.0 / textureSize(Sampler, 0);
    vec4 Color = texture(Sampler, TexCoord) /sqrt(pi*sigma);
    int h = 5;
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

void main()
{
    outColor = vec4(0.0f,0.0f,0.0f,0.0f);
    //outColor += blur(blurSampler,fragTexCoord);
    outColor += texture(blurSampler,fragTexCoord);
}
