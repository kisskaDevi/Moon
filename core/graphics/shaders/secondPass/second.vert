#version 450

layout(set = 0, binding = 4) uniform GlobalUniformBuffer
{
    mat4 view;
    mat4 proj;
    vec4 eyePosition;
} global;

layout(set = 1, binding = 0) uniform LightBufferObject
{
    mat4 proj;
    mat4 view;
    mat4 projView;
    vec4 position;
    vec4 lightColor;
    vec4 lightProp;
}light;

layout(location = 0)	out vec4 eyePosition;
layout(location = 1)	out vec2 fragTexCoord;
layout(location = 2)	out vec4 glPosition;

layout(location = 3)	out vec3 lightPosition;
layout(location = 4)	out vec4 lightColor;
layout(location = 5)	out vec4 lightProp;
layout(location = 6)	out mat4 lightProjView;
layout(location = 10)	out mat4 projview;

vec3 vertex[5];
int index[18] = int[](0,4,1,0,1,2,0,2,3,0,3,4,4,2,1,2,4,3);

void main()
{
    lightPosition   = light.position.xyz;
    lightColor	    = light.lightColor;
    lightProp	    = light.lightProp;
    lightProjView   = light.projView;

    projview	= global.proj * global.view;
    eyePosition = global.eyePosition;

    vec3 n =  - normalize(vec3(light.view[0][2],light.view[1][2],light.view[2][2]));
    vec3 u =    normalize(vec3(light.view[0][0],light.view[1][0],light.view[2][0]));
    vec3 v =    normalize(vec3(light.view[0][1],light.view[1][1],light.view[2][1]));

    float far  = -light.proj[3][2]/(-light.proj[2][2]-1.0f);
    float h = far/light.proj[1][1];
    float w = light.proj[1][1]/light.proj[0][0]*h;

    vertex[0] = lightPosition;
    vertex[1] = lightPosition + far*n + w*u +  h*v;
    vertex[2] = lightPosition + far*n + w*u -  h*v;
    vertex[3] = lightPosition + far*n - w*u -  h*v;
    vertex[4] = lightPosition + far*n - w*u +  h*v;

    float gh = far/global.proj[1][1];
    float gw = global.proj[1][1]/global.proj[0][0]*h;

    glPosition = projview * vec4(vertex[index[gl_VertexIndex]],1.0f);
    fragTexCoord = glPosition.xy;
    fragTexCoord.x /= gw;
    fragTexCoord.y /= -gh;
    gl_Position = glPosition;
    glPosition = vec4(vertex[index[gl_VertexIndex]],1.0f);
}
