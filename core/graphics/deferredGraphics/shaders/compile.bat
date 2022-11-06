glslc.exe base/base.vert				    -o base/basevert.spv
glslc.exe base/base.frag				    -o base/basefrag.spv
glslc.exe SpotLightingPass/SpotLighting.vert		    -o SpotLightingPass/SpotLightingVert.spv
glslc.exe SpotLightingPass/SpotLighting.frag		    -o SpotLightingPass/SpotLightingFrag.spv
glslc.exe SpotLightingPass/SpotLightingScattering.frag      -o SpotLightingPass/SpotLightingScatteringFrag.spv
glslc.exe SpotLightingPass/SpotLightingAmbient.vert	    -o SpotLightingPass/AmbientSpotLightingVert.spv
glslc.exe SpotLightingPass/SpotLightingAmbient.frag	    -o SpotLightingPass/AmbientSpotLightingFrag.spv
glslc.exe shadow/shadowMapShader.vert                       -o shadow/shad.spv
glslc.exe customFilter/customFilter.vert		    -o customFilter/customFilterVert.spv
glslc.exe customFilter/customFilter.frag		    -o customFilter/customFilterFrag.spv
glslc.exe postProcessing/postProcessingShader.vert	    -o postProcessing/postProcessingVert.spv
glslc.exe postProcessing/postProcessingShader.frag	    -o postProcessing/postProcessingFrag.spv
glslc.exe skybox/skybox.vert				    -o skybox/skyboxVert.spv
glslc.exe skybox/skybox.frag				    -o skybox/skyboxFrag.spv
glslc.exe stencil/secondStencil.vert			    -o stencil/secondstencilvert.spv
glslc.exe stencil/secondStencil.frag			    -o stencil/secondstencilfrag.spv
glslc.exe sslr/SSLR.vert				    -o sslr/sslrVert.spv
glslc.exe sslr/SSLR.frag				    -o sslr/sslrFrag.spv
glslc.exe ssao/SSAO.vert				    -o ssao/ssaoVert.spv
glslc.exe ssao/SSAO.frag				    -o ssao/ssaoFrag.spv
glslc.exe combiner/combiner.vert                            -o combiner/combinerVert.spv
glslc.exe combiner/combiner.frag                            -o combiner/combinerFrag.spv
glslc.exe gaussianBlur/xBlur.vert                           -o gaussianBlur/xBlurVert.spv
glslc.exe gaussianBlur/xBlur.frag                           -o gaussianBlur/xBlurFrag.spv
glslc.exe gaussianBlur/yBlur.vert               	    -o gaussianBlur/yBlurVert.spv
glslc.exe gaussianBlur/yBlur.frag               	    -o gaussianBlur/yBlurFrag.spv

pause
