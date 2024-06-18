#include "deferredAttachments.h"

namespace moon::deferredGraphics {

const moon::utils::Attachments& GBufferAttachments::operator[](uint32_t index) const {
    return *(&position + index);
}

const moon::utils::Attachments& DeferredAttachments::operator[](uint32_t index) const {
    return *(&image + index);
}

std::vector<VkClearValue> DeferredAttachments::clearValues() const {
    std::vector<VkClearValue> value;
    for (uint32_t i = 0; i < static_cast<uint32_t>(size()); i++) {
        value.push_back((*this)[i].clearValue());
    }
    return value;
}
}
