#version 450
#define MAX_NODE_COUNT 256

layout(set = 3, binding = 0) uniform sampler2D baseColorTexture;
layout(set = 3, binding = 1) uniform sampler2D metallicRoughnessTexture;
layout(set = 3, binding = 2) uniform sampler2D normalTexture;
layout(set = 3, binding = 3) uniform sampler2D occlusionTexture;
layout(set = 3, binding = 4) uniform sampler2D emissiveTexture;

layout(location = 0)	in vec3 position;
layout(location = 1)	in vec2 UV0;
layout(location = 2)	in vec2 UV1;
layout(location = 3)	in vec4 outColor;
layout(location = 4)	in float depth;
layout(location = 5)	in vec4 glPosition;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outBaseColor;
layout(location = 3) out vec4 outEmissiveTexture;

struct Material
{
    vec4 baseColorFactor;
    vec4 emissiveFactor;
    vec4 diffuseFactor;
    vec4 specularFactor;
    float workflow;
    int baseColorTextureSet;
    int physicalDescriptorTextureSet;
    int normalTextureSet;
    int occlusionTextureSet;
    int emissiveTextureSet;
    float metallicFactor;
    float roughnessFactor;
    float alphaMask;
    float alphaMaskCutoff;
    int index;
    int firstIndex;
};

layout(set = 0, binding = 2) uniform MaterialUniformBufferObject
{
    Material ubo[MAX_NODE_COUNT];
} material;

layout(set = 0, binding = 3) buffer StorageBuffer
{
    vec4 mousePosition;
    int number;
    float depth;
} storage;


layout (push_constant) uniform MaterialPC
{
    int normalTextureSet;
    int number;
} materialPC;

void main()
{
    int number = materialPC.number;

    outPosition = vec4(position,1.0f);
    //outBaseColor = texture(baseColorTexture, UV0);
    outBaseColor = vec4(0.0f);
    outNormal = vec4(0.0f,0.0f,0.0f,0.0f);
    outEmissiveTexture = outColor+texture(emissiveTexture, UV0);

    outPosition.a = depth;

    if(storage.depth>glPosition.z/glPosition.w){
	if(abs(glPosition.x-storage.mousePosition.x)<0.002&&abs(glPosition.y-storage.mousePosition.y)<0.002){
	    storage.number = number;
	    storage.depth = glPosition.z/glPosition.w;
	}
    }
}

