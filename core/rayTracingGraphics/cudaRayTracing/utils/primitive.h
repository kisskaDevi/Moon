#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include "hitable/hitable.h"
#include "utils/devicep.h"

#include <vector>
#include <algorithm>

namespace cuda {

struct primitive{
    devicep<hitable> hit;
    box bbox;

    box getBox() const {
        return bbox;
    }
};

inline void sortByBox(std::vector<const primitive*>::iterator begin, std::vector<const primitive*>::iterator end, const box& bbox){
    const vec4f limits = bbox.max - bbox.min;
    std::sort(begin, end, [i = limits.maxValueIndex(3)](const primitive* a, const primitive* b){
        return a->getBox().min[i] < b->getBox().min[i];
    });
}

inline std::vector<hitable*> extractHitables(const std::vector<const primitive*>& storage){
    std::vector<hitable*> hitables;
    for(const auto& p : storage){
        hitables.push_back(p->hit());
    }
    return hitables;
}

}

#endif // PRIMITIVE_H
