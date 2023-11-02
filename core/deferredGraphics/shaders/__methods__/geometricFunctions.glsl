#ifndef GEOMETRIC_FUNCTIONS
#define GEOMETRIC_FUNCTIONS

bool checkZeroNormal(const in vec3 normal) {
    return normal.x == 0.0 && normal.y == 0.0 && normal.z == 0.0;
}

float getAspect(const in mat4 proj){
    return - proj[1][1] / proj[0][0];
}

vec3 getDirection(const in mat4 view){
    return - normalize(vec3(view[0][2], view[1][2], view[2][2]));
}

bool outsideSpotCondition(const in mat4 proj, const in mat4 view, const in float type, const in vec3 position) {
    vec4 coordinates = view * vec4(position, 1.0) * vec4(getAspect(proj), 1.0, -1.0, 1.0);

    return type == 0.0 
    ? sqrt(coordinates.x * coordinates.x + coordinates.y * coordinates.y) >= coordinates.z
    : abs(coordinates.x) >= abs(coordinates.z) || abs(coordinates.y) >= abs(coordinates.z);
}

#endif