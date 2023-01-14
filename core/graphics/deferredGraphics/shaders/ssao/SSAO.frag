#version 450

layout(set = 0, binding = 1) uniform sampler2D position;
layout(set = 0, binding = 2) uniform sampler2D normal;
layout(set = 0, binding = 3) uniform sampler2D Sampler;
layout(set = 0, binding = 4) uniform sampler2D depth;
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


    return occlusion;
}

void main()
{
    //outColor = vec4(SSAO());
    outColor = vec4(0.0f);
}
