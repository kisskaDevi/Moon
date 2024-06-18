#ifndef DEFERREDATTACHMENTS_H
#define DEFERREDATTACHMENTS_H

#include "attachments.h"

namespace moon::deferredGraphics {

struct GBufferAttachments{
    moon::utils::Attachments         position;
    moon::utils::Attachments         normal;
    moon::utils::Attachments         color;
    moon::utils::Attachments         depth;

    GBufferAttachments() = default;
    GBufferAttachments(const GBufferAttachments& other) = delete;
    GBufferAttachments& operator=(const GBufferAttachments& other) = delete;
    GBufferAttachments(GBufferAttachments&& other) = default;
    GBufferAttachments& operator=(GBufferAttachments&& other) = default;

    const moon::utils::Attachments& operator[](uint32_t index) const;

    constexpr static uint32_t size() {return 4;}
    constexpr static uint32_t positionIndex() {return 0;}
    constexpr static uint32_t normalIndex() {return 1;}
    constexpr static uint32_t colorIndex() {return 2;}
    constexpr static uint32_t depthIndex() {return 3;}
};

struct DeferredAttachments{
    moon::utils::Attachments         image;
    moon::utils::Attachments         blur;
    moon::utils::Attachments         bloom;
    GBufferAttachments               GBuffer;

    DeferredAttachments() = default;
    DeferredAttachments(const DeferredAttachments& other) = delete;
    DeferredAttachments& operator=(const DeferredAttachments& other) = delete;
    DeferredAttachments(DeferredAttachments&& other) = default;
    DeferredAttachments& operator=(DeferredAttachments&& other) = default;

    const moon::utils::Attachments& operator[](uint32_t index) const;

    constexpr static uint32_t size() {return 3 + GBufferAttachments::size();}
    constexpr static uint32_t imageIndex() {return 0;}
    constexpr static uint32_t blurIndex() {return 1;}
    constexpr static uint32_t bloomIndex() {return 2;}
    constexpr static uint32_t GBufferOffset() {return 3;}

    std::vector<VkClearValue> clearValues() const;
};

}
#endif // DEFERREDATTACHMENTS_H
