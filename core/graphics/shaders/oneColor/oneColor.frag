#version 450

const float pi = 3.141592653589793f;
const float PBR_WORKFLOW_METALLIC_ROUGHNESS = 0.0;
const float PBR_WORKFLOW_SPECULAR_GLOSINESS = 1.0f;
const float c_MinRoughness = 0.04;

layout(set = 0, binding = 1)	uniform samplerCube samplerCubeMap;

layout(set = 0, binding = 2) buffer StorageBuffer
{
    vec4 mousePosition;
    int number;
    float depth;
} storage;

layout (push_constant) uniform MaterialPC
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
    int number;
} materialPC;


layout(set = 3, binding = 0) uniform sampler2D baseColorTexture;
layout(set = 3, binding = 1) uniform sampler2D metallicRoughnessTexture;
layout(set = 3, binding = 2) uniform sampler2D normalTexture;
layout(set = 3, binding = 3) uniform sampler2D occlusionTexture;
layout(set = 3, binding = 4) uniform sampler2D emissiveTexture;

layout(location = 0)	in vec4 position;
layout(location = 1)	in vec3 normal;
layout(location = 2)	in vec2 UV0;
layout(location = 3)	in vec2 UV1;
layout(location = 4)	in vec3 tangent;
layout(location = 5)	in vec3 bitangent;
layout(location = 6)	in vec4 color;
layout(location = 7)	in vec4 eyePosition;
layout(location = 8)	in float depth;
layout(location = 9)	in vec4 glPosition;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outBaseColor;
layout(location = 3) out vec4 outEmissiveTexture;

vec3		getNormal();
vec4		SRGBtoLINEAR(vec4 srgbIn);
float		convertMetallic(vec3 diffuse, vec3 specular, float maxSpecular);

void main()
{
    outPosition = position;
    outBaseColor = color;
    outNormal = vec4(materialPC.normalTextureSet > -1 ? getNormal() : normal, 0.0f);
    outEmissiveTexture = texture(emissiveTexture, UV0);

    if(outBaseColor.a!=1.0f){
	discard;
    }

//    vec3 I = normalize(position.xyz - eyePosition.xyz);
//    vec3 R = reflect(I, outNormal.xyz);
//    vec4 reflection = texture(samplerCubeMap, R);
//    outBaseColor = vec4(max(outBaseColor.r,reflection.r),max(outBaseColor.g,reflection.g),max(outBaseColor.b,reflection.b), outBaseColor.a);

    float perceptualRoughness;
    float metallic;
    vec4 baseColor;

    int number = materialPC.number;

    if (materialPC.workflow == PBR_WORKFLOW_METALLIC_ROUGHNESS)
    {
	// Metallic and Roughness material properties are packed together
	// In glTF, these factors can be specified by fixed scalar values
	// or from a metallic-roughness map
	perceptualRoughness = materialPC.roughnessFactor;
	metallic	    = materialPC.metallicFactor;
	if (materialPC.physicalDescriptorTextureSet > -1) {
	        // Roughness is stored in the 'g' channel, metallic is stored in the 'b' channel.
	        // This layout intentionally reserves the 'r' channel for (optional) occlusion map data
	        vec4 mrSample = texture(metallicRoughnessTexture,UV0);
		perceptualRoughness = mrSample.g * perceptualRoughness;
		metallic = mrSample.b * metallic;
	} else {
	        perceptualRoughness = clamp(perceptualRoughness, c_MinRoughness, 1.0);
		metallic = clamp(metallic, 0.0, 1.0);
	}

	// The albedo may be defined from a base texture or a flat color
	if (materialPC.baseColorTextureSet > -1) {
	        baseColor = SRGBtoLINEAR(outBaseColor) * materialPC.baseColorFactor;
	} else {
	        baseColor = materialPC.baseColorFactor;
	}
    }

    if (materialPC.workflow == PBR_WORKFLOW_SPECULAR_GLOSINESS)
    {
	// Values from specular glossiness workflow are converted to metallic roughness
	if (materialPC.physicalDescriptorTextureSet > -1) {
	        perceptualRoughness = 1.0 - texture(metallicRoughnessTexture,UV0).a;
	} else {
	        perceptualRoughness = 0.0;
	}

	vec4 diffuse = SRGBtoLINEAR(outBaseColor);
	vec3 specular = SRGBtoLINEAR(texture(metallicRoughnessTexture,UV0)).rgb;
	float maxSpecular = max(max(specular.r, specular.g), specular.b);

	// Convert metallic value from specular glossiness inputs
	metallic = convertMetallic(diffuse.rgb, specular, maxSpecular);

	const float epsilon = 1e-6;

	vec3 baseColorDiffusePart = diffuse.rgb * ((1.0 - maxSpecular) / (1 - c_MinRoughness) / max(1 - metallic, epsilon)) * materialPC.diffuseFactor.rgb;
	vec3 baseColorSpecularPart = specular - (vec3(c_MinRoughness) * (1 - metallic) * (1 / max(metallic, epsilon))) * materialPC.specularFactor.rgb;
	baseColor = vec4(mix(baseColorDiffusePart, baseColorSpecularPart, metallic * metallic), diffuse.a);
    }


    outPosition.a = depth;
    outBaseColor = vec4(baseColor.xyz,perceptualRoughness);
    outNormal.a = metallic;
    if (materialPC.occlusionTextureSet > -1) {
	    float ao = texture(occlusionTexture,UV0).r;
	    outEmissiveTexture.a = ao;
    }else{
	outEmissiveTexture.a = 1.0f;
    }

    if(storage.depth>glPosition.z/glPosition.w){
	if(abs(glPosition.x-storage.mousePosition.x)<0.002&&abs(glPosition.y-storage.mousePosition.y)<0.002){
	    storage.number = number;
	    storage.depth = glPosition.z/glPosition.w;
	}
    }
}

vec3 getNormal()
{
    vec3 tangentNormal = normalize(texture(normalTexture, materialPC.normalTextureSet == 0 ? UV0 : UV1).xyz * 2.0f - 1.0f);

    mat3 TBN = mat3(tangent, bitangent, normal);

    return normalize(TBN * tangentNormal);
}

vec4 SRGBtoLINEAR(vec4 srgbIn)
{
        #ifdef MANUAL_SRGB
            #ifdef SRGB_FAST_APPROXIMATION
                vec3 linOut = pow(srgbIn.xyz,vec3(2.2));
            #else //SRGB_FAST_APPROXIMATION
                vec3 bLess = step(vec3(0.04045),srgbIn.xyz);
		vec3 linOut = mix( srgbIn.xyz/vec3(12.92), pow((srgbIn.xyz+vec3(0.055))/vec3(1.055),vec3(2.4)), bLess );
            #endif //SRGB_FAST_APPROXIMATION
	    return vec4(linOut,srgbIn.w);
        #else //MANUAL_SRGB
        return srgbIn;
        #endif //MANUAL_SRGB
}

float convertMetallic(vec3 diffuse, vec3 specular, float maxSpecular)
{
        float perceivedDiffuse = sqrt(0.299 * diffuse.r * diffuse.r + 0.587 * diffuse.g * diffuse.g + 0.114 * diffuse.b * diffuse.b);
	float perceivedSpecular = sqrt(0.299 * specular.r * specular.r + 0.587 * specular.g * specular.g + 0.114 * specular.b * specular.b);
	if (perceivedSpecular < c_MinRoughness) {
	        return 0.0;
	}
	float a = c_MinRoughness;
	float b = perceivedDiffuse * (1.0 - maxSpecular) / (1.0 - c_MinRoughness) + perceivedSpecular - 2.0 * c_MinRoughness;
	float c = c_MinRoughness - perceivedSpecular;
	float D = max(b * b - 4.0 * a * c, 0.0);
	return clamp((-b + sqrt(D)) / (2.0 * a), 0.0, 1.0);
}
