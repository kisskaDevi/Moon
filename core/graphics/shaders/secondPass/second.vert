#version 450
#define MAX_LIGHT_SOURCES 20

layout(set = 0, binding = 6) uniform GlobalUniformBuffer
{
    mat4 view;
    mat4 proj;
    vec4 eyePosition;
} global;

layout(location = 0)	out vec4 eyePosition;
layout(location = 1)	out vec2 fragTexCoord;
layout(location = 2)	out vec4 glPosition;

layout(location = 3)	out vec3 lightPosition;
layout(location = 4)	out vec4 lightColor;
layout(location = 5)	out int type;
layout(location = 6)	out mat4 lightProjView;
layout(location = 10)	out mat4 projview;

struct LightBufferObject
{
    mat4 proj;
    mat4 view;
    mat4 projView;
    vec4 position;
    vec4 lightColor;
    int type;
    int enableShadow;
    int enableScattering;
};

layout (push_constant) uniform LightPushConst
{
    int number;
} lightPC;

layout(set = 0, binding = 4) uniform LightUniformBufferObject
{
    LightBufferObject ubo[MAX_LIGHT_SOURCES];
} light;

vec3 vertex[5];
int index[18] = int[](0,4,1,0,1,2,0,2,3,0,3,4,4,2,1,2,4,3);

void main()
{
    int i = lightPC.number;

    lightPosition   = light.ubo[i].position.xyz;
    lightColor	    = light.ubo[i].lightColor;
    type	    = light.ubo[i].type;
    lightProjView	    = light.ubo[i].projView;

    mat4 proj	= global.proj;
    projview	= global.proj * global.view;
    eyePosition = global.eyePosition;


    vec3 n =  - normalize(vec3(light.ubo[i].view[0][2],light.ubo[i].view[1][2],light.ubo[i].view[2][2]));
    vec3 u =    normalize(vec3(light.ubo[i].view[0][0],light.ubo[i].view[1][0],light.ubo[i].view[2][0]));
    vec3 v =    normalize(vec3(light.ubo[i].view[0][1],light.ubo[i].view[1][1],light.ubo[i].view[2][1]));

    float far  = -light.ubo[i].proj[3][2]/(-light.ubo[i].proj[2][2]-1.0f);
    float h = far/light.ubo[i].proj[1][1];
    float w = light.ubo[i].proj[1][1]/light.ubo[i].proj[0][0]*h;

    vertex[0] = lightPosition;
    vertex[1] = lightPosition + far*n + w*u +  h*v;
    vertex[2] = lightPosition + far*n + w*u -  h*v;
    vertex[3] = lightPosition + far*n - w*u -  h*v;
    vertex[4] = lightPosition + far*n - w*u +  h*v;

    glPosition = projview * vec4(vertex[index[gl_VertexIndex]],1.0f);
    fragTexCoord = glPosition.xy;
    fragTexCoord.x /= w;
    fragTexCoord.y /= -h;
    gl_Position = glPosition;
    glPosition = vec4(vertex[index[gl_VertexIndex]],1.0f);
}
