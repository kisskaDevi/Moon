#version 450
#define MANUAL_SRGB 1
#define pi 3.141592653589793f

layout (push_constant) uniform PC
{
    float minAmbientFactor;
}pc;

layout(location = 0)	in vec4 eyePosition;
layout(location = 1)	in vec2 fragTexCoord;
layout(location = 2)	in vec4 glPosition;

layout(location = 3)	flat in vec3 lightPosition;
layout(location = 4)	flat in vec4 lightColor;
layout(location = 5)	flat in vec4 lightProp;
layout(location = 6)	flat in mat4 lightProjView;
layout(location = 10)	flat in mat4 projview;

layout(input_attachment_index = 0, binding = 0) uniform subpassInput inPositionTexture;
layout(input_attachment_index = 1, binding = 1) uniform subpassInput inNormalTexture;
layout(input_attachment_index = 2, binding = 2) uniform subpassInput inBaseColorTexture;
layout(input_attachment_index = 3, binding = 3) uniform subpassInput inEmissiveTexture;
layout(input_attachment_index = 4, binding = 4) uniform subpassInput inDepthTexture;

layout(set = 1, binding = 1) uniform sampler2D shadowMap;
layout(set = 1, binding = 2) uniform sampler2D lightTexture;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outBlur;
layout(location = 2) out vec4 outBloom;

struct attenuation{
    float C;
    float L;
    float Q;
};

struct Vector{
    vec3 eyeDirection;
    vec3 lightDirection;
    vec3 normal;
    vec3 reflect;
    vec3 H;
}vector;

vec4 position;
vec4 normal;
vec4 baseColorTexture;
vec4 emissiveTexture;

vec4 fragLightPosition;
vec4 textureLightColor;
float depthMap;

float type;
float lightPowerFactor;
float lightDropFactor;

//===================================================functions====================================================================//
bool		outsideSpotCondition(vec3 lightSpaceNDC, float type);
float		shadowFactor();
vec4		SRGBtoLINEAR(vec4 srgbIn);
vec3		specularReflection(vec3 specularEnvironmentR0, vec3 specularEnvironmentR90, float VdotH);
float		geometricOcclusion(float NdotL, float NdotV, float r);
float		microfacetDistribution(float NdotH, float alphaRoughness);
vec3		diffuse(vec3 diffuseColor);
attenuation	getK(float distance);

//===================================================outImages====================================================================//

vec4 PBR(vec4 outColor)
{
    fragLightPosition = lightProjView * vec4(position.xyz,1.0f);
    textureLightColor = vec4(0.0f,0.0f,0.0f,1.0f);

    float metallic = subpassLoad(inNormalTexture).a;
    float perceptualRoughness = subpassLoad(inPositionTexture).a;
    vec3 diffuseColor;
    vec4 baseColor = baseColorTexture;
    baseColor = SRGBtoLINEAR(baseColor);

    vec3 f0 = vec3(0.04);

    diffuseColor = baseColor.rgb * (vec3(1.0) - f0);
    diffuseColor *= 1.0 - metallic;

    float alphaRoughness = perceptualRoughness * perceptualRoughness;

    vec3 specularColor = mix(f0, baseColor.rgb, metallic);

    // Compute reflectance.
    float reflectance = max(max(specularColor.r, specularColor.g), specularColor.b);
    float reflectance90 = clamp(reflectance * 25.0, 0.0, 1.0);
    vec3 specularEnvironmentR0 = specularColor.rgb;
    vec3 specularEnvironmentR90 = vec3(1.0, 1.0, 1.0) * reflectance90;


        float len = length(lightPosition - position.xyz);
	attenuation K = getK(len);
	float lightDrop = K.C+K.L*len+K.Q*len*len;

	lightDrop *= lightDropFactor;

	float ShadowFactor = shadowFactor();
	if(ShadowFactor>pc.minAmbientFactor){
	    float lightPower = ShadowFactor*lightPowerFactor;

	    vector.eyeDirection	    = normalize(eyePosition.xyz - position.xyz);
	    vector.lightDirection   = normalize(lightPosition - position.xyz);
	    vector.normal	    = normal.xyz;
	    vector.reflect	    = -normalize(reflect(vector.eyeDirection, vector.normal));
	    vector.H		    = normalize(vector.eyeDirection + vector.lightDirection);

	    vec3  F = specularReflection(specularEnvironmentR0, specularEnvironmentR90, clamp(dot(vector.eyeDirection, vector.H), 0.0, 1.0));
	    float G = geometricOcclusion(clamp(dot(vector.normal, vector.lightDirection), 0.001, 1.0), clamp(abs(dot(vector.normal, vector.eyeDirection)), 0.001, 1.0), alphaRoughness);
	    float D = microfacetDistribution(clamp(dot(vector.normal, vector.H), 0.0, 1.0), alphaRoughness);

	    // Calculation of analytical lighting contribution
	    vec3 diffuseContrib = (1.0f - F) * diffuse(diffuseColor);
	    vec3 specContrib = F * G * D / (4.0 * clamp(dot(vector.normal, vector.lightDirection), 0.001, 1.0) * clamp(abs(dot(vector.normal, vector.eyeDirection)), 0.001, 1.0));
	    // Obtain final intensity as reflectance (BRDF) scaled by the energy of the light (cosine law)
	    vec3 color = clamp(dot(vector.normal, vector.lightDirection), 0.001, 1.0) * vec3(max(lightColor.x,textureLightColor.x),max(lightColor.y,textureLightColor.y),max(lightColor.z,textureLightColor.z)) * (diffuseContrib + specContrib);

	    const float u_OcclusionStrength = 1.0f;
	    float ao = emissiveTexture.a;
	    color = mix(color, color * ao, u_OcclusionStrength);

	    color = lightPower*color/lightDrop;
	    outColor = vec4(color, baseColor.a);
	}


//	//=========== Area Light ===========//
//	    vec4 rayColor = vec4(0.0f,0.0f,0.0f,0.0f);
//	    vec3 pm = findMirrorVector(eyePosition.xyz,position.xyz,normal.xyz);
//	    rayColor += planeIntersection(pm,position.xyz,lightPosition,1.0f,lightProj,lightView)*vec4(max(lightColor.x,textureLightColor.x),max(lightColor.y,textureLightColor.y),max(lightColor.z,textureLightColor.z),1.0f)/lightDrop;
//	    rayColor *= metallic;
//	    outColor = vec4(max(rayColor.r,outColor.r),max(rayColor.g,outColor.g),max(rayColor.b,outColor.b), baseColor.a);

	return outColor;
}

vec4 LightScattering(int steps)
{
    vec4 outScatteringColor = vec4(0.0f,0.0f,0.0f,0.0f);
    vec3 eyeDirection = glPosition.xyz - eyePosition.xyz;
    vec3 dstep = eyeDirection/steps;

    //    float dstep = 0.3f;
    //    eyeDirection = normalize(eyeDirection);
    //	  vec3 pointOfScattering = eyePosition.xyz + dstep * step * eyeDirection;

    int insideCounter = 0;

    for(int step=1;step<=steps;step++){
	vec3 pointOfScattering = eyePosition.xyz + step * dstep;
	float cameraViewZ = projview[0][2]*pointOfScattering.x + projview[1][2]*pointOfScattering.y + projview[2][2]*pointOfScattering.z + projview[3][2];
	float cameraViewW = projview[0][3]*pointOfScattering.x + projview[1][3]*pointOfScattering.y + projview[2][3]*pointOfScattering.z + projview[3][3];
	if(depthMap>cameraViewZ/cameraViewW){
	    vec4 lightView = lightProjView * vec4(pointOfScattering, 1.0f);
	    vec3 lightSpaceNDC = lightView.xyz/lightView.w;
	    if(!outsideSpotCondition(lightSpaceNDC,type))
	    {
		vec2 shadowMapCoord = lightSpaceNDC.xy * 0.5f + 0.5f;
		if(lightSpaceNDC.z<texture(shadowMap, shadowMapCoord.xy).x)
		{
		    vec4 color = texture(lightTexture, shadowMapCoord.xy);
		    color = vec4(max(color.x,lightColor.x),max(color.y,lightColor.y),max(color.z,lightColor.z),max(color.a,lightColor.a));

		    float len = length(lightPosition - pointOfScattering);

		    outScatteringColor += vec4(color.xyz,0.0f)*exp(-0.2f*(len));
		    insideCounter++;
		}
	    }
	}
    }
    outScatteringColor /= insideCounter*4;
    return outScatteringColor;
}

//===================================================main====================================================================//

void main()
{
    position = subpassLoad(inPositionTexture);
    normal = subpassLoad(inNormalTexture);
    baseColorTexture = subpassLoad(inBaseColorTexture);
    emissiveTexture = subpassLoad(inEmissiveTexture);
    depthMap = subpassLoad(inDepthTexture).r;

    outColor = vec4(0.0f,0.0f,0.0f,0.0f);
    outBlur = vec4(0.0f,0.0f,0.0f,0.0f);
    outBloom = vec4(0.0f,0.0f,0.0f,0.0f);

    type = lightProp.x;
    lightPowerFactor = lightProp.y;
    lightDropFactor = lightProp.z;

    if(normal.x==0.0f&&normal.y==0.0f&&normal.z==0.0f)	outColor = SRGBtoLINEAR(baseColorTexture);
    else						outColor = PBR(outColor);

        outBlur  += LightScattering(100);
	outColor += SRGBtoLINEAR(emissiveTexture);
	outBloom += SRGBtoLINEAR(emissiveTexture);

    if(outColor.x>0.95f&&outColor.y>0.95f&&outColor.y>0.95f)	outBloom += outColor;
    else							outBloom += vec4(0.0f,0.0f,0.0f,1.0f);

}

//===========================================================================================================================//

bool outsideSpotCondition(vec3 lightSpaceNDC, float type)
{
    if(type==0.0f)
	return sqrt(lightSpaceNDC.x*lightSpaceNDC.x + lightSpaceNDC.y*lightSpaceNDC.y) >= lightSpaceNDC.z;
    else
	return abs(lightSpaceNDC.x) > 1.0f || abs(lightSpaceNDC.y) > 1.0f || abs(lightSpaceNDC.z) > 1.0f;
}

float shadowFactor()
{
    vec3 lightSpaceNDC = fragLightPosition.xyz;
    lightSpaceNDC /= fragLightPosition.w;

    float shadowSample = 0.0f;

        vec2 shadowMapCoord = lightSpaceNDC.xy * 0.5f + 0.5f;

	int n = 8; int maxNoise = 1;
	float dang = 2.0f*pi/n; float dnoise = 0.001f;
	for(int j=0;j<n;j++)
	{
	    for(float noise = dnoise; noise<dnoise*(maxNoise+1); noise+=dnoise)
	    {
		vec2 dx = vec2(noise*cos(j*dang), noise*sin(j*dang));
		shadowSample += lightSpaceNDC.z-texture(shadowMap, shadowMapCoord.xy + dx).x > 0.001f ? 1.0f : 0.0f;
	    }
	}
	shadowSample /= maxNoise*n;

	textureLightColor += texture(lightTexture, shadowMapCoord.xy);

    float factor = pc.minAmbientFactor;
    if(!outsideSpotCondition(lightSpaceNDC,type))
	if(1.0f - shadowSample>pc.minAmbientFactor)
	    factor = 1.0f - shadowSample;

    return factor;
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

// The following equation models the Fresnel reflectance term of the spec equation (aka F())
// Implementation of fresnel from [4], Equation 15
vec3 specularReflection(vec3 specularEnvironmentR0, vec3 specularEnvironmentR90, float VdotH)
{
        return specularEnvironmentR0 + (specularEnvironmentR90 - specularEnvironmentR0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
}

// This calculates the specular geometric attenuation (aka G()),
// where rougher material will reflect less light back to the viewer.
// This implementation is based on [1] Equation 4, and we adopt their modifications to
// alphaRoughness as input as originally proposed in [2].
float geometricOcclusion(float NdotL, float NdotV, float r)
{
        float attenuationL = 2.0 * NdotL / (NdotL + sqrt(r * r + (1.0 - r * r) * (NdotL * NdotL)));
	float attenuationV = 2.0 * NdotV / (NdotV + sqrt(r * r + (1.0 - r * r) * (NdotV * NdotV)));
	return attenuationL * attenuationV;
}

// The following equation(s) model the distribution of microfacet normals across the area being drawn (aka D())
// Implementation from "Average Irregularity Representation of a Roughened Surface for Ray Reflection" by T. S. Trowbridge, and K. P. Reitz
// Follows the distribution function recommended in the SIGGRAPH 2013 course notes from EPIC Games [1], Equation 3.
float microfacetDistribution(float NdotH, float alphaRoughness)
{
        float roughnessSq = alphaRoughness * alphaRoughness;
	float f = (NdotH * roughnessSq - NdotH) * NdotH + 1.0;
	return roughnessSq / (pi * f * f);
}

// Basic Lambertian diffuse
// Implementation from Lambert's Photometria https://archive.org/details/lambertsphotome00lambgoog
// See also [1], Equation 1
vec3 diffuse(vec3 diffuseColor)
{
    return diffuseColor / pi;
}

attenuation getK(float distance)
{
    attenuation res;
    res.C = 1.0f;
    res.L = 0.0866f*exp(-0.00144f*distance);
    res.Q = 0.0283f*exp(-0.00289f*distance);
    return res;
}

//vec3 findMirrorVector(vec3 pv, vec3 p0, vec3 n)
//{
//    vec3 v = p0 - pv;
//    return pv + 2*n*dot(n,v);
//}

//float sphereIntersection(vec3 pm, vec3 p0, vec3 pl, float R)
//{
//    vec3 a = p0 - pm;
//    vec3 l = pm - pl;

//    float b = dot(a,l)/dot(a,a);
//    float c = (dot(l,l) - R*R)/dot(a,a);

//    if(b*b-c>=0)
//	return 1.0f;
//    else
//	return 0.0f;
//}

//float planeIntersection(vec3 pm, vec3 p0, vec3 pl, float R, mat4 P, mat4 V)
//{
//    vec3 u = normalize(vec3(V[0][0],V[1][0],V[2][0]));
//    vec3 v = normalize(vec3(V[0][1],V[1][1],V[2][1]));
//    vec3 n = -normalize(vec3(V[0][2],V[1][2],V[2][2]));

//    vec3 ps = pl + R*n;
//    if(dot(ps-pm,n)<=0)
//    {
//	float t = dot(n,pm-ps)/dot(n,pm-p0);
//	float h = -R/P[1][1];
//	float w = R/P[0][0];
//	vec3 qs = (pm - ps) - (pm - p0)*t;

//	u = h*u;
//	v = w*v;

//	bool xcond = -h<=dot(qs,u)/sqrt(dot(u,u))&&dot(qs,u)/sqrt(dot(u,u))<=h;
//	bool ycond = -w<=dot(qs,v)/sqrt(dot(v,v))&&dot(qs,v)/sqrt(dot(v,v))<=w;

//	if(xcond&&ycond)
//	    return 1.0f;
//	else
//	    return 0.0f;
//    }
//    else{
//	return 0.0f;
//    }
//}
