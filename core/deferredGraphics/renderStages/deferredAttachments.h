#ifndef DEFERREDATTACHMENTS_H
#define DEFERREDATTACHMENTS_H

#include "attachments.h"

namespace moon::deferredGraphics {

struct GBufferAttachments{
    moon::utils::Attachments         position;
    moon::utils::Attachments         normal;
    moon::utils::Attachments         color;
    moon::utils::Attachments         depth;

    GBufferAttachments();
    GBufferAttachments(const GBufferAttachments& other);
    GBufferAttachments& operator=(const GBufferAttachments& other);

    inline const moon::utils::Attachments& operator[](uint32_t index) const {
        return *(&position + index);
    }

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
    GBufferAttachments  GBuffer;

    DeferredAttachments();
    DeferredAttachments(const DeferredAttachments& other);
    DeferredAttachments& operator=(const DeferredAttachments& other);

    inline const moon::utils::Attachments& operator[](uint32_t index) const {
        return *(&image + index);
    }

    void deleteAttachment(VkDevice device);
    void deleteSampler(VkDevice device);

    constexpr static uint32_t size() {return 3 + GBufferAttachments::size();}
    constexpr static uint32_t imageIndex() {return 0;}
    constexpr static uint32_t blurIndex() {return 1;}
    constexpr static uint32_t bloomIndex() {return 2;}
    constexpr static uint32_t GBufferOffset() {return 3;}
};

}
#endif // DEFERREDATTACHMENTS_H
