#version 450

#include "../__methods__/defines.glsl"

layout(set = 0, binding = 4) uniform GlobalUniformBuffer {
    mat4 view;
    mat4 proj;
    vec4 eyePosition;
} global;

layout(set = 1, binding = 0) uniform LightBufferObject {
    mat4 proj;
    mat4 view;
    mat4 projView;
    vec4 position;
    vec4 color;
    vec4 prop;
} light;

layout(location = 0) out vec4 eyePosition;

vec3 vertex[5];
int index[18] = int[](
        0,4,1,
        0,1,2,
        0,2,3,
        0,3,4,
        4,2,1,
        2,4,3
);

void main() {
    eyePosition = global.eyePosition;

    vec3 u =   normalize(vec3(light.view[0][0], light.view[1][0], light.view[2][0]));
    vec3 v =   normalize(vec3(light.view[0][1], light.view[1][1], light.view[2][1]));
    vec3 n = - normalize(vec3(light.view[0][2], light.view[1][2], light.view[2][2]));

    float far = light.proj[3][2] / (light.proj[2][2] + 1.0);
    float h = - far / light.proj[1][1];
    float w = - far / light.proj[0][0];

    vec3 center = light.position.xyz + far * n;
    vec3 x = w * u;
    vec3 y = h * v;

    vertex[0] = light.position.xyz;
    vertex[1] = center + x + y;
    vertex[2] = center + x - y;
    vertex[3] = center - x - y;
    vertex[4] = center - x + y;

    gl_Position = global.proj * global.view * vec4(vertex[index[gl_VertexIndex]], 1.0);
}
