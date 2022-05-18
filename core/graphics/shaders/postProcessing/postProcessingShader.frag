#version 450

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

layout(set = 0, binding = 5) uniform sampler2D bloomSampler[8];

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

const float pi = 3.141592653589793f;

vec4 blur(sampler2D Sampler, vec2 TexCoord)
{
    float sigma = 2.0 * textureSize(Sampler, 0).y;
    vec2 textel = 2.0 / textureSize(Sampler, 0);
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

//======================== SSR ==========================================//
const float stepScale = 1.0;
const float maxSteps = 20;
const int numBinarySearchSteps = 30;

vec4 BinarySearch(vec3 reflectDir, vec3 hitCoord, mat4 view, mat4 proj);
vec4 RayMarch(vec3 rayStep, vec3 hitCoord, mat4 view, mat4 proj);
vec3 fresnelSchlick(float cosTheta, vec3 F0);
//======================================================================//

void main()
{
    outColor = vec4(0.0f,0.0f,0.0f,0.0f);

    vec3 pointPosition	= texture(position, fragTexCoord).xyz;
    vec3 pointNormal	= texture(normal,   fragTexCoord).xyz;
    vec3 pointOfView	= global.eyePosition.xyz;
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
		outColor += vec4(Fresnel,1.0f)*texture(Sampler,result.xy);
	}
    }

    outColor += texture(Sampler,fragTexCoord);
    outColor += blur(blurSampler,fragTexCoord);

    vec4 bloomColor = vec4(0.0f);
    float blit = 1.0f/1.5f;
    for(int i=0;i<8;i++){
	vec2 coord = fragTexCoord*blit;
	bloomColor += texture(bloomSampler[i],coord)*exp(0.01*i*i);
	blit/=1.5f;
    }
    bloomColor /= 3.0f;
    outColor += bloomColor;
}

vec4 RayMarch(vec3 reflectDir, vec3 hitCoord, mat4 view, mat4 proj)
{
    vec3 rayStep = reflectDir;
    rayStep *= stepScale;

    vec4 depthProjectedCoord;
    vec4 rayProjectedCoord;
    float delta;
    float deltaz;

    for(int i = 0; i < maxSteps; i++)
    {
	hitCoord += rayStep;

	rayProjectedCoord = proj * view * vec4(hitCoord, 1.0f);
	deltaz = rayProjectedCoord.z;
	rayProjectedCoord /= rayProjectedCoord.w;
	rayProjectedCoord.xy = rayProjectedCoord.xy * 0.5f + 0.5f;

	vec3 depthCoord = texture(position, rayProjectedCoord.xy).xyz;
	depthProjectedCoord = proj * view * vec4(depthCoord, 1.0f);
	deltaz -= depthProjectedCoord.z;
	depthProjectedCoord /= depthProjectedCoord.w;

	delta = depthProjectedCoord.z - rayProjectedCoord.z;

	if(abs(deltaz)<1.0f){
	    if(delta <= 0.0){
		vec4 result;
		if(dot(reflectDir, texture(normal,rayProjectedCoord.xy).xyz)>=-0.3f){
		    result = vec4(rayProjectedCoord.xy, 0.0f, 0.0);
		}else{
		    result = BinarySearch(rayStep, hitCoord, view, proj);
		}
		return result;
	    }
	}

    }

    return vec4(rayProjectedCoord.xy, 0.0f, 0.0f);
}

vec4 BinarySearch(vec3 rayStep, vec3 hitCoord, mat4 view, mat4 proj)
{
    float delta;

    vec4 depthProjectedCoord;
    vec4 rayProjectedCoord;
    vec3 dir = rayStep;
    float deltaz;

    for(int i = 0; i < numBinarySearchSteps; i++)
    {
	rayProjectedCoord = proj * view * vec4(hitCoord, 1.0f);
	deltaz = rayProjectedCoord.z;
	rayProjectedCoord /= rayProjectedCoord.w;
	rayProjectedCoord.xy = rayProjectedCoord.xy * 0.5f + 0.5f;

	vec3 depthCoord = texture(position, rayProjectedCoord.xy).xyz;
	depthProjectedCoord = proj * view * vec4(depthCoord, 1.0f);
	deltaz -= depthProjectedCoord.z;
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

    if(deltaz>0.05f)
	rayProjectedCoord.w=0.0f;

    return rayProjectedCoord;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

