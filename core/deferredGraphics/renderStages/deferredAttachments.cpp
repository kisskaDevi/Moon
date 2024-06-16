#include "deferredAttachments.h"

namespace moon::deferredGraphics {

const moon::utils::Attachments& GBufferAttachments::operator[](uint32_t index) const {
    return *(&position + index);
}

const moon::utils::Attachments& DeferredAttachments::operator[](uint32_t index) const {
    return *(&image + index);
}
}
