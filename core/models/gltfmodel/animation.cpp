#include "../gltfmodel.h"

#include <iostream>

bool gltfModel::hasAnimation(uint32_t frameIndex) const {
    return instances[instances.size() > frameIndex ? frameIndex : 0].animations.size() > 0;
}

float gltfModel::animationStart(uint32_t frameIndex, uint32_t index) const {
    return instances[frameIndex].animations[index].start;
}

float gltfModel::animationEnd(uint32_t frameIndex, uint32_t index) const {
    return instances[frameIndex].animations[index].end;
}

void gltfModel::loadAnimations(tinygltf::Model& gltfModel)
{
    for(auto& instance: instances){
        for (tinygltf::Animation &anim : gltfModel.animations) {
            Animation animation{};

            // Samplers
            for (auto &samp : anim.samplers) {
                Animation::AnimationSampler sampler{};

                if (samp.interpolation == "LINEAR") {
                    sampler.interpolation = Animation::AnimationSampler::InterpolationType::LINEAR;
                }
                if (samp.interpolation == "STEP") {
                    sampler.interpolation = Animation::AnimationSampler::InterpolationType::STEP;
                }
                if (samp.interpolation == "CUBICSPLINE") {
                    sampler.interpolation = Animation::AnimationSampler::InterpolationType::CUBICSPLINE;
                }

                // Read sampler input time values
                {
                    const tinygltf::Accessor &accessor = gltfModel.accessors[samp.input];
                    const tinygltf::BufferView &bufferView = gltfModel.bufferViews[accessor.bufferView];
                    const tinygltf::Buffer &buffer = gltfModel.buffers[bufferView.buffer];

                    assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);

                    const void *dataPtr = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
                    const float *buf = static_cast<const float*>(dataPtr);
                    for (size_t index = 0; index < accessor.count; index++) {
                        sampler.inputs.push_back(buf[index]);
                    }

                    for (const auto& input: sampler.inputs) {
                        animation.start = std::min(input, animation.start);
                        animation.end = std::max(input, animation.end);
                    }
                }

                // Read sampler output T/R/S values
                {
                    const tinygltf::Accessor &accessor = gltfModel.accessors[samp.output];
                    const tinygltf::BufferView &bufferView = gltfModel.bufferViews[accessor.bufferView];
                    const tinygltf::Buffer &buffer = gltfModel.buffers[bufferView.buffer];

                    assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);

                    const void *dataPtr = &buffer.data[accessor.byteOffset + bufferView.byteOffset];

                    switch (accessor.type) {
                        case TINYGLTF_TYPE_VEC3: {
                            const glm::vec3 *buf = static_cast<const glm::vec3*>(dataPtr);
                            for (size_t index = 0; index < accessor.count; index++) {
                                sampler.outputsVec4.push_back(glm::vec4(buf[index], 0.0f));
                            }
                            break;
                        }
                        case TINYGLTF_TYPE_VEC4: {
                            const glm::vec4 *buf = static_cast<const glm::vec4*>(dataPtr);
                            for (size_t index = 0; index < accessor.count; index++) {
                                sampler.outputsVec4.push_back(buf[index]);
                            }
                            break;
                        }
                        default: {
                            std::cout << "unknown type" << std::endl;
                            break;
                        }
                    }
                }

                animation.samplers.push_back(sampler);
            }

            // Channels
            for (auto &source: anim.channels)
            {
                Animation::AnimationChannel channel{};

                if (source.target_path == "rotation") {
                    channel.path = Animation::AnimationChannel::PathType::ROTATION;
                }
                if (source.target_path == "translation") {
                    channel.path = Animation::AnimationChannel::PathType::TRANSLATION;
                }
                if (source.target_path == "scale") {
                    channel.path = Animation::AnimationChannel::PathType::SCALE;
                }
                if (source.target_path == "weights") {
                    std::cout << "weights not yet supported, skipping channel" << std::endl;
                    continue;
                }
                channel.samplerIndex = source.sampler;
                if (channel.node = nodeFromIndex(source.target_node, instance.nodes); channel.node) {
                    animation.channels.push_back(channel);
                }
            }

            instance.animations.push_back(animation);
        }
    }
}

void gltfModel::updateAnimation(uint32_t frameIndex, uint32_t index, float time)
{
    if (instances[frameIndex].animations.empty()) {
        std::cout << ".glTF does not contain animation." << std::endl;
        return;
    }
    if (index > static_cast<uint32_t>(instances[frameIndex].animations.size()) - 1) {
        std::cout << "No animation with index " << index << std::endl;
        return;
    }
    Animation &animation = instances[frameIndex].animations[index];

    bool updated = false;
    for (auto& channel : animation.channels) {
        Animation::AnimationSampler &sampler = animation.samplers[channel.samplerIndex];
        if (sampler.inputs.size() > sampler.outputsVec4.size()) {
            continue;
        }

        for (size_t i = 0; i < sampler.inputs.size() - 1; i++) {
            if ((time >= sampler.inputs[i]) && (time <= sampler.inputs[i + 1])) {
                float u = std::max(0.0f, time - sampler.inputs[i]) / (sampler.inputs[i + 1] - sampler.inputs[i]);
                if (u <= 1.0f) {
                    switch (channel.path) {
                        case Animation::AnimationChannel::PathType::TRANSLATION: {
                            glm::vec4 trans = glm::mix(sampler.outputsVec4[i], sampler.outputsVec4[i + 1], u);
                            channel.node->translation = glm::vec3(trans);
                            break;
                        }
                        case Animation::AnimationChannel::PathType::SCALE: {
                            glm::vec4 trans = glm::mix(sampler.outputsVec4[i], sampler.outputsVec4[i + 1], u);
                            channel.node->scale = glm::vec3(trans);
                            break;
                        }
                        case Animation::AnimationChannel::PathType::ROTATION: {
                            glm::quat q1;
                            q1.x = sampler.outputsVec4[i].x;
                            q1.y = sampler.outputsVec4[i].y;
                            q1.z = sampler.outputsVec4[i].z;
                            q1.w = sampler.outputsVec4[i].w;
                            glm::quat q2;
                            q2.x = sampler.outputsVec4[i + 1].x;
                            q2.y = sampler.outputsVec4[i + 1].y;
                            q2.z = sampler.outputsVec4[i + 1].z;
                            q2.w = sampler.outputsVec4[i + 1].w;
                            channel.node->rotation = glm::normalize(glm::slerp(q1, q2, u));
                            break;
                        }
                    }
                    updated = true;
                }
            }
        }
    }
    if (updated) {
        for (auto &node : instances[frameIndex].nodes) {
            node->update();
        }
    }
}

void gltfModel::changeAnimation(uint32_t frameIndex, uint32_t oldIndex, uint32_t newIndex, float startTime, float time, float changeAnimationTime)
{
    Animation &animationOld = instances[frameIndex].animations[oldIndex];
    Animation &animationNew = instances[frameIndex].animations[newIndex];

    bool updated = false;
    for (auto& channel : animationOld.channels) {
        Animation::AnimationSampler &samplerOld = animationOld.samplers[channel.samplerIndex];
        Animation::AnimationSampler &samplerNew = animationNew.samplers[channel.samplerIndex];
        if (samplerOld.inputs.size() > samplerOld.outputsVec4.size())
            continue;

        for (size_t i = 0; i < samplerOld.inputs.size(); i++) {
            if ((startTime >= samplerOld.inputs[i]) && (time <= samplerOld.inputs[i]+changeAnimationTime)) {
                float u = std::max(0.0f, time - startTime) / changeAnimationTime;
                if (u <= 1.0f) {
                    switch (channel.path) {
                        case Animation::AnimationChannel::PathType::TRANSLATION: {
                            glm::vec4 trans = glm::mix(samplerOld.outputsVec4[i], samplerNew.outputsVec4[0], u);
                            channel.node->translation = glm::vec3(trans);
                            break;
                        }
                        case Animation::AnimationChannel::PathType::SCALE: {
                            glm::vec4 trans = glm::mix(samplerOld.outputsVec4[i], samplerNew.outputsVec4[0], u);
                            channel.node->scale = glm::vec3(trans);
                            break;
                        }
                        case Animation::AnimationChannel::PathType::ROTATION: {
                            glm::quat q1;
                            q1.x = samplerOld.outputsVec4[i].x;
                            q1.y = samplerOld.outputsVec4[i].y;
                            q1.z = samplerOld.outputsVec4[i].z;
                            q1.w = samplerOld.outputsVec4[i].w;
                            glm::quat q2;
                            q2.x = samplerNew.outputsVec4[0].x;
                            q2.y = samplerNew.outputsVec4[0].y;
                            q2.z = samplerNew.outputsVec4[0].z;
                            q2.w = samplerNew.outputsVec4[0].w;
                            channel.node->rotation = glm::normalize(glm::slerp(q1, q2, u));
                            break;
                        }
                    }
                    updated = true;
                }
            }
        }
    }
    if (updated) {
        for (auto &node : instances[frameIndex].nodes) {
            node->update();
        }
    }
}
