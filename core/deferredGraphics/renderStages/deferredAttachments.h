#ifndef DEFERREDATTACHMENTS_H
#define DEFERREDATTACHMENTS_H

#include "attachments.h"

struct GBufferAttachments{
    attachments         position;
    attachments         normal;
    attachments         color;
    attachments         depth;

    GBufferAttachments();
    GBufferAttachments(const GBufferAttachments& other);
    GBufferAttachments& operator=(const GBufferAttachments& other);

    inline const attachments& operator[](uint32_t index) const {
        return *(&position + index);
    }

    constexpr static uint32_t size() {return 4;}
    constexpr static uint32_t positionIndex() {return 0;}
    constexpr static uint32_t normalIndex() {return 1;}
    constexpr static uint32_t colorIndex() {return 2;}
    constexpr static uint32_t depthIndex() {return 3;}
};

struct DeferredAttachments{
    attachments         image;
    attachments         blur;
    attachments         bloom;
    GBufferAttachments  GBuffer;

    DeferredAttachments();
    DeferredAttachments(const DeferredAttachments& other);
    DeferredAttachments& operator=(const DeferredAttachments& other);

    inline const attachments& operator[](uint32_t index) const {
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

#endif // DEFERREDATTACHMENTS_H
