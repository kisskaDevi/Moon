#ifndef GEOMETRIC_FUNCTIONS
#define GEOMETRIC_FUNCTIONS

bool checkZeroNormal(vec3 normal) {
    return normal.x == 0.0 && normal.y == 0.0 && normal.z == 0.0;
}

#endif