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

//vec2 positions[6] = vec2[](
//    vec2(-1.0f, -1.0f),
//    vec2( 1.0f, -1.0f),
//    vec2( 1.0f,  1.0f),
//    vec2(1.0f, 1.0f),
//    vec2(-1.0f, 1.0f),
//    vec2( -1.0f,  -1.0f)
//);

vec2 step[6] = vec2[](
    vec2(0.0f, 0.0f),
    vec2( 1.0f, 0.0f),
    vec2( 1.0f,  1.0f),
    vec2(1.0f, 1.0f),
    vec2(0.0f, 1.0f),
    vec2( 0.0f,  0.0f)
);

vec2 fragCoord[6] = vec2[](
    vec2(0.0f, 0.0f),
    vec2(1.0f, 0.0f),
    vec2(1.0f, 1.0f),
    vec2(1.0f, 1.0f),
    vec2(0.0f, 1.0f),
    vec2(0.0f, 0.0f)
);

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

void main()
{
    eyePosition = global.eyePosition;

    int xsteps = 1;
    int ysteps = 1;

    float x0 = -1.0f;
    float y0 = -1.0f;
    float dx = 2.0f/float(xsteps);
    float dy = 2.0f/float(ysteps);

    float fx0 = 0.0f;
    float fy0 = 0.0f;
    float fdx = 1.0f/float(xsteps);
    float fdy = 1.0f/float(ysteps);

    int arrayIndex = gl_VertexIndex % 6;
    int tileNumber = (gl_VertexIndex - arrayIndex)/6;
    int tileX = tileNumber % xsteps;
    int tileY = (tileNumber - tileX) / ysteps;

    float x = x0 + tileX*dx + step[arrayIndex].x * dx;
    float y = y0 + tileY*dy + step[arrayIndex].y * dy;

    float fx = fx0 + tileX*fdx + step[arrayIndex].x * fdx;
    float fy = fy0 + tileY*fdy + step[arrayIndex].y * fdy;

    fragTexCoord = vec2(fx,fy);

    glPosition = vec4(vec2(x,y),0.0, 1.0);
    gl_Position = vec4(vec2(x,y),0.0, 1.0);
}
