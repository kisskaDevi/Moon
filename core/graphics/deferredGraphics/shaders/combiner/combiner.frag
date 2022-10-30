#version 450

layout(set = 0, binding = 0) uniform sampler2D Samplers[4];
layout(set = 0, binding = 1) uniform sampler2D depthSamplers[4];
layout(set = 0, binding = 2) uniform sampler2D depth;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = vec4(0.0f);

    for(int i=0;i<4;i++){
        if(texture(depthSamplers[i],fragTexCoord).r<=texture(depth,fragTexCoord).r){
            outColor += texture(Samplers[i],fragTexCoord);
        }
    }
}
