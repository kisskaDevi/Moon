#ifndef OUTSIDE_SPOT_CONDITION_GLSL
#define OUTSIDE_SPOT_CONDITION_GLSL

bool outsideSpotCondition(vec3 coordinates, float type) {
    if(type == 0.0) {
        return sqrt(coordinates.x * coordinates.x + coordinates.y * coordinates.y) >= coordinates.z;
    } else {
        return abs(coordinates.x) >= abs(coordinates.z) || abs(coordinates.y) >= abs(coordinates.z);
    }
}

#endif