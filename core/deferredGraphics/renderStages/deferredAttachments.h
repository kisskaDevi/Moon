#ifndef DEFERREDATTACHMENTS_H
#define DEFERREDATTACHMENTS_H

#include "attachments.h"

namespace moon::deferredGraphics {

struct GBufferAttachments{
    utils::Attachments position;
    utils::Attachments normal;
    utils::Attachments color;
    utils::Attachments depth;

    const utils::Attachments& operator[](uint32_t index) const {
        return *(&position + index);
    }

    constexpr static uint32_t size() {return 4;}
    constexpr static uint32_t positionIndex() {return 0;}
    constexpr static uint32_t normalIndex() {return 1;}
    constexpr static uint32_t colorIndex() {return 2;}
    constexpr static uint32_t depthIndex() {return 3;}
};

struct DeferredAttachments{
    utils::Attachments image;
    utils::Attachments blur;
    utils::Attachments bloom;
    GBufferAttachments GBuffer;

    const utils::Attachments& operator[](uint32_t index) const {
        return *(&image + index);
    }

    constexpr static uint32_t size() {return 3 + GBufferAttachments::size();}
    constexpr static uint32_t imageIndex() {return 0;}
    constexpr static uint32_t blurIndex() {return 1;}
    constexpr static uint32_t bloomIndex() {return 2;}
    constexpr static uint32_t GBufferOffset() {return 3;}

    std::vector<VkClearValue> DeferredAttachments::clearValues() const {
        std::vector<VkClearValue> value;
        for (uint32_t i = 0; i < size(); i++) {
            value.push_back((*this)[i].clearValue());
        }
        return value;
    }
};

}
#endif // DEFERREDATTACHMENTS_H
