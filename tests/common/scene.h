#ifndef SCENE_H
#define SCENE_H

#include <cstdint>

class scene
{
public:
    virtual ~scene(){};

    virtual void create(uint32_t WIDTH, uint32_t HEIGHT) = 0;
    virtual void resize(uint32_t WIDTH, uint32_t HEIGHT) = 0;
    virtual void updateFrame(uint32_t frameNumber, float frameTime) = 0;
    virtual void destroy() = 0;
};

#endif // SCENE_H
