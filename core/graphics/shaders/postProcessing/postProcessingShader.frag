#version 450
#define bloomCount 8

layout(set = 0, binding = 0) uniform sampler2D Sampler;
layout(set = 0, binding = 1) uniform sampler2D blurSampler;
layout(set = 0, binding = 2) uniform sampler2D position;
layout(set = 0, binding = 3) uniform sampler2D normal;
layout(set = 0, binding = 4) uniform GlobalUniformBuffer
{
    mat4 view;
    mat4 proj;
    vec4 eyePosition;
} global;
layout(set = 0, binding = 5) uniform sampler2D bloomSampler[bloomCount];

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
vec3 pointPosition	= texture(position, fragTexCoord).xyz;
vec3 pointNormal	= texture(normal,   fragTexCoord).xyz;
vec3 pointOfView	= global.eyePosition.xyz;

mat4 proj = global.proj;
mat4 view = global.view;
mat4 projview = proj * view;

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
    bloomColor /= 3.0f;

    return bloomColor;
}

//======================== SSLR ==========================================//
const float stepScale = 1.0;
const float maxSteps = 20;
const int numBinarySearchSteps = 30;

vec4 BinarySearch(vec3 reflectDir, vec3 hitCoord, mat4 view, mat4 proj);
vec4 RayMarch(vec3 rayStep, vec3 hitCoord, mat4 view, mat4 proj);
vec3 fresnelSchlick(float cosTheta, vec3 F0);
vec4 SSLR();
//========================================================================//

//======================== SSAO ==========================================//
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
//========================================================================//

void main()
{
    outColor = vec4(0.0f,0.0f,0.0f,0.0f);

    outColor += vec4(texture(Sampler,fragTexCoord).xyz,0.0f);
//    outColor += vec4(0.0f,texture(Sampler,fragTexCoord).y,0.0f,0.0f);
//    outColor += vec4(texture(Sampler,fragTexCoord+vec2(0.005f)).x,0.0f,0.0f,0.0f);
//    outColor += vec4(0.0f,0.0f,texture(Sampler,fragTexCoord-vec2(0.005f)).z,0.0f);
    outColor += blur(blurSampler,fragTexCoord);
    outColor += bloom();

//    outColor += SSLR();

//    float ssao = SSAO();
//    outColor = vec4(ssao);
}

vec4 SSLR()
{
    vec4 returnColor = vec4(0.0f);
    vec3 reflectDir	= normalize(reflect(pointPosition-pointOfView,pointNormal));
    mat4 view		= global.view;
    mat4 proj		= global.proj;

    float metallic = texture(normal,fragTexCoord).a;
    vec3 F0 = vec3(0.004);
    F0      = mix(F0, texture(Sampler, fragTexCoord).xyz, metallic);
    vec3 Fresnel = fresnelSchlick(max(dot(normalize(pointNormal), normalize(pointPosition-pointOfView)), 0.0), F0);

    if(pointNormal.x!=0.0f&&pointNormal.y!=0.0f&&pointNormal.z!=0.0f){
	vec4 result = RayMarch(reflectDir,pointPosition,view,proj);
	if(result.w==1.0f){
	    if(result.x<1.0f&&result.x>0.0f&&result.y<1.0f&&result.y>0.0f)
		returnColor += vec4(Fresnel,1.0f)*texture(Sampler,result.xy);
	}
    }
    return returnColor;
}

vec4 RayMarch(vec3 reflectDir, vec3 hitCoord, mat4 view, mat4 proj)
{
    vec3 rayStep = reflectDir;
    rayStep *= stepScale;

    vec4 depthProjectedCoord;
    vec4 rayProjectedCoord;
    float deltaz;

    for(int i = 0; i < maxSteps; i++)
    {
	hitCoord += rayStep;

	rayProjectedCoord = projview * vec4(hitCoord, 1.0f);
	deltaz = rayProjectedCoord.z;
	rayProjectedCoord /= rayProjectedCoord.w;
	rayProjectedCoord.xy = rayProjectedCoord.xy * 0.5f + 0.5f;
	deltaz -= texture(position, rayProjectedCoord.xy).a;

	if(abs(deltaz)<1.0f){
	    if(-deltaz <= 0.0){
		vec4 result;
		if(dot(reflectDir, texture(normal,rayProjectedCoord.xy).xyz)>=-0.3f)
		    result = vec4(rayProjectedCoord.xy, 0.0f, 0.0);
		else
		    result = BinarySearch(rayStep, hitCoord, view, proj);

		return result;
	    }
	}

    }

    return vec4(rayProjectedCoord.xy, 0.0f, 0.0f);
}

vec4 BinarySearch(vec3 rayStep, vec3 hitCoord, mat4 view, mat4 proj)
{
    vec4 depthProjectedCoord;
    vec4 rayProjectedCoord;
    vec3 dir = rayStep;
    float deltaz;

    for(int i = 0; i < numBinarySearchSteps; i++)
    {
	rayProjectedCoord = projview * vec4(hitCoord, 1.0f);
	deltaz = rayProjectedCoord.z - texture(position, rayProjectedCoord.xy).a;
	rayProjectedCoord /= rayProjectedCoord.w;
	rayProjectedCoord.xy = rayProjectedCoord.xy * 0.5f + 0.5f;
	deltaz -= texture(position, rayProjectedCoord.xy).a;

	dir *= 0.5f;
	if(-deltaz > 0.0f)  hitCoord += dir;
	else		    hitCoord -= dir;
    }

    if(deltaz>0.05f)
	rayProjectedCoord.w=0.0f;

    return rayProjectedCoord;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

