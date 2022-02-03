#version 450

layout(set = 0, binding = 1)	uniform samplerCube samplerCubeMap;

layout(location = 0)	in vec3 inUVW;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outBaseColor;
layout(location = 3) out vec4 outMetallicRoughness;
layout(location = 4) out vec4 outOcclusion;
layout(location = 5) out vec4 outEmissiveTexture;

//===================================================functions====================================================================//

vec4 SRGBtoLINEAR(vec4 srgbIn);

//===================================================main====================================================================//

void main()
{
    outPosition = vec4(inUVW,1.0f);
    outBaseColor = texture(samplerCubeMap, inUVW);
    outMetallicRoughness = vec4(0.0f);
    outNormal = vec4(0.0f);
    outOcclusion = vec4(0.0f);
    outEmissiveTexture = vec4(0.0f);

    outPosition.a = 3.0f;
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
	return vec4(linOut,srgbIn.w);;
        #else //MANUAL_SRGB
        return srgbIn;
        #endif //MANUAL_SRGB
}
