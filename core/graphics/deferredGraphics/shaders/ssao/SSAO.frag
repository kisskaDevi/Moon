#version 450

layout(set = 0, binding = 1) uniform sampler2D position;
layout(set = 0, binding = 2) uniform sampler2D normal;
layout(set = 0, binding = 3) uniform sampler2D Sampler;
layout(set = 0, binding = 0) uniform GlobalUniformBuffer
{
    mat4 view;
    mat4 proj;
    vec4 eyePosition;
} global;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

const float pi = 3.141592653589793f;
vec3 pointPosition	= texture(position, fragTexCoord).xyz;
vec3 pointNormal	= texture(normal,   fragTexCoord).xyz;
vec3 pointOfView	= global.eyePosition.xyz;

mat4 proj = global.proj;
mat4 view = global.view;
mat4 projview = proj * view;

float SSAO()
{
    float occlusion = 0.0f;

    float bias = 0.05f;

    float xn = pointNormal.x;
    float yn = pointNormal.y;
    float zn = pointNormal.z;
    float eta = 1.0f;
    float xi = - (zn*zn+xn*xn+xn*yn*eta)/(xn*yn+eta*(yn*yn+zn*zn));
    vec3 tangent = normalize(vec3(1.0f,eta,-(xn+yn*eta)/zn));
    vec3 bitangent = normalize(vec3(1.0f,xi,-(xn+yn*xi)/zn));

    int steps = 4;
    float R0 = 0.0f, Rn = 0.2f, dR = (Rn-R0)/(steps-1);
    float phi0 = 0.2f*pi, phin = 1.8f*pi, dphi = (phin-phi0)/(steps-1);
    float theta0 = 0.1f*pi, thetan = 0.5f*pi, dtheta = (thetan-theta0)/(steps-1);

    float counter = 0.0f;
    for(int i=0;i<steps;i++)
    {
	float R = R0 + i*dR;
	for(int j=0;j<steps;j++)
	{
	    float phi = phi0 + j*dphi;
	    for(int k=0;k<steps;k++)
	    {
		float theta = theta0 + k*dtheta;

		float x = R*sin(theta)*cos(phi);
		float y = R*sin(theta)*sin(phi);
		float z = R*cos(theta);
		vec3 samplePos = pointPosition + x*tangent + y*bitangent + z*pointNormal;

		vec4 offset = vec4(samplePos, 1.0);
		offset      = projview * offset;
		vec2 offsetCoord  = offset.xy/offset.w * 0.5f + 0.5f;

		float sampleDepth = texture(position, offsetCoord.xy).a;

		float delta = sampleDepth - (offset.z + bias);

		if(delta < 0.0f){
		    float range = length(samplePos - texture(position, offsetCoord.xy).xyz);
		    if(range<0.02f){
			occlusion += 1.0f;
			counter += 1.0f;
		    }
		}else{
		    counter += 1.0f;
		}
	    }
	}
    }
    occlusion = 1.0 - (occlusion / counter);
    return occlusion;
}

void main()
{
    outColor = vec4(SSAO());
    //outColor = vec4(0.0f);
}
