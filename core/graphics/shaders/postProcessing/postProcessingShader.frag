#version 450

layout(set = 0, binding = 0) uniform sampler2D Sampler;
layout(set = 0, binding = 1) uniform sampler2D bloomSampler;

layout(set = 0, binding = 2) uniform sampler2D position;
layout(set = 0, binding = 3) uniform sampler2D normal;

layout(set = 0, binding = 4) uniform GlobalUniformBuffer
{
    mat4 view;
    mat4 proj;
    vec4 eyePosition;
} global;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

const float pi = 3.141592653589793f;

vec4 blur(sampler2D Sampler, vec2 TexCoord)
{
    float sigma = 2.0 * textureSize(Sampler, 0).y;
    vec2 textel = 1.0 / textureSize(Sampler, 0);
    vec3 Color = texture(Sampler, TexCoord).xyz /sqrt(pi*sigma);
    int h = 20;
    float Norm = 1.0f/sqrt(pi*sigma);
    for(int i=-h;i<h+1;i+=2)
    {
	float I1 = Norm * exp( -(i*textel.y*i*textel.y)/sigma);
	float I2 = Norm * exp( -((i+1)*textel.y*(i+1)*textel.y)/sigma);
	float y = (I1*i+I2*(i+1))*textel.y/(I1+I2);
	float I = Norm * exp( -(y*y)/sigma);
	Color += texture(Sampler, TexCoord + vec2(0.0f,y) ).xyz * I;
	Color += texture(Sampler, TexCoord - vec2(0.0f,y) ).xyz * I;
    }
    return vec4(Color, 1.0);
}

vec4 radialBlur(sampler2D bloomSampler, vec2 TexCoord)
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

//========================
const float stepScale = 0.5;
const float maxSteps = 30;
const int numBinarySearchSteps = 10;
const float eps = 0.000001;

vec4 BinarySearch(vec3 reflectDir, vec3 hitCoord, mat4 view, mat4 proj);
vec4 RayMarch(vec3 rayStep, vec3 hitCoord, mat4 view, mat4 proj);
//=======================

void main()
{
    outColor = vec4(0.0f,0.0f,0.0f,0.0f);

    vec3 pointPosition	= texture(position, fragTexCoord).xyz;
    vec3 pointNormal	= texture(normal,   fragTexCoord).xyz;
    vec3 pointOfView	= global.eyePosition.xyz;
    vec3 reflectDir	= normalize(reflect(pointPosition-pointOfView,pointNormal));
    mat4 view		= global.view;
    mat4 proj		= global.proj;

    vec4 result = RayMarch(reflectDir,pointPosition,view,proj);
    if(result.w==1.0f){
	if(result.x<1.0f&&result.x>0.0f&&result.y<1.0f&&result.y>0.0f)
	outColor += texture(Sampler,result.xy);
    }

    outColor += texture(Sampler,fragTexCoord);
    outColor += blur(bloomSampler,fragTexCoord);
}

vec4 RayMarch(vec3 reflectDir, vec3 hitCoord, mat4 view, mat4 proj)
{
    vec3 rayStep = reflectDir;
    rayStep *= stepScale;

    vec4 depthProjectedCoord;
    vec4 rayProjectedCoord;
    float delta;

    for(int i = 0; i < maxSteps; i++)
    {
	hitCoord += rayStep;

	rayProjectedCoord = proj * view * vec4(hitCoord, 1.0f);
	rayProjectedCoord /= rayProjectedCoord.w;
	rayProjectedCoord.xy = rayProjectedCoord.xy * 0.5f + 0.5f;

	vec3 depthCoord = texture(position, rayProjectedCoord.xy).xyz;
	depthProjectedCoord = proj * view * vec4(depthCoord, 1.0f);
	depthProjectedCoord /= depthProjectedCoord.w;

	delta = depthProjectedCoord.z - rayProjectedCoord.z;

	if(delta <= 0.0)
	{
	    if(dot(reflectDir, texture(normal,rayProjectedCoord.xy).xyz)>=-0.3f){
		return vec4(rayProjectedCoord.xy, 0.0f, 0.0);
	    }else{
		return BinarySearch(rayStep, hitCoord, view, proj);
	    }
	}

    }

    return vec4(rayProjectedCoord.xy, 0.0f, 0.0);
}

vec4 BinarySearch(vec3 rayStep, vec3 hitCoord, mat4 view, mat4 proj)
{
    float delta;

    vec4 depthProjectedCoord;
    vec4 rayProjectedCoord;
    vec3 dir = rayStep;

    for(int i = 0; i < numBinarySearchSteps; i++)
    {
	rayProjectedCoord = proj * view * vec4(hitCoord, 1.0f);
	rayProjectedCoord /= rayProjectedCoord.w;
	rayProjectedCoord.xy = rayProjectedCoord.xy * 0.5f + 0.5f;

	vec3 depthCoord = texture(position, rayProjectedCoord.xy).xyz;
	depthProjectedCoord = proj * view * vec4(depthCoord, 1.0f);
	depthProjectedCoord /= depthProjectedCoord.w;

	delta = depthProjectedCoord.z - rayProjectedCoord.z;

	dir *= 0.5f;
	if(delta > 0.0f)
	    hitCoord += dir;
	else
	    hitCoord -= dir;
    }

    rayProjectedCoord = proj * view * vec4(hitCoord, 1.0f);
    rayProjectedCoord /= rayProjectedCoord.w;
    rayProjectedCoord.xy = rayProjectedCoord.xy * 0.5f + 0.5f;

    vec3 depthCoord = texture(position, rayProjectedCoord.xy).xyz;
    depthProjectedCoord = proj * view * vec4(depthCoord, 1.0f);
    depthProjectedCoord /= depthProjectedCoord.w;

    delta = abs(depthProjectedCoord.z/rayProjectedCoord.z - 1.0f);

    if(delta<eps){
	rayProjectedCoord.w = 1.0f;
    }else{
	rayProjectedCoord.w = 0.0f;
    }

    return rayProjectedCoord;
}
