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

//======================== SSLR ==========================================//
const float stepScale = 1.0;
const float maxSteps = 40;
const int numBinarySearchSteps = 20;

vec4 BinarySearch(vec3 reflectDir, vec3 hitCoord, mat4 view, mat4 proj);
vec4 RayMarch(vec3 rayStep, vec3 hitCoord, mat4 view, mat4 proj);
vec3 fresnelSchlick(float cosTheta, vec3 F0);
vec4 SSLR();

void main()
{
    outColor = vec4(0.0f,0.0f,0.0f,0.0f);

    outColor += SSLR();
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

    if(!(pointNormal.x==0.0f&&pointNormal.y==0.0f&&pointNormal.z==0.0f)){
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

	if(abs(deltaz)<3.0f){
	    if(deltaz >= 0.0){
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

    //if(deltaz>0.05f)
        //rayProjectedCoord.w=0.0f;

    return rayProjectedCoord;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
