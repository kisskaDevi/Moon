#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include "hitable/hitable.h"
#include "utils/devicep.h"

#include <vector>
#include <algorithm>

namespace cuda::rayTracing {

struct Primitive{
    Devicep<Hitable> hit;
    box bbox;

    box getBox() const {
        return bbox;
    }
};

inline void sortByBox(std::vector<const Primitive*>::iterator begin, std::vector<const Primitive*>::iterator end, const box& bbox){
    const vec4f limits = bbox.max - bbox.min;
    std::sort(begin, end, [i = limits.maxValueIndex(3)](const Primitive* a, const Primitive* b){
        return a->getBox().min[i] < b->getBox().min[i];
    });
}

inline std::vector<Hitable*> extractHitables(const std::vector<const Primitive*>& storage){
    std::vector<Hitable*> hitables;
    for(const auto& p : storage){
        hitables.push_back(p->hit());
    }
    return hitables;
}

}
#endif // PRIMITIVE_H
