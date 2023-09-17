#include <iostream>
#include <math.h>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>

#include "triangle.h"
#include "sphere.h"
#include "hitableContainer.h"

#include "object.h"

enum sign
{
    minus,
    plus
};

std::vector<vertex> createBoxVertexBuffer(vec4 scale, vec4 translate, sign normalSign, properties props, std::vector<vec4> colors) {
    float plus = normalSign == sign::plus ? 1.0f : -1.0f, minus = -plus;
    vec4 v[8] =
    {
        scale * vec4(-1.0f, -1.0f, -1.0f, 1.0f) + translate,
        scale * vec4(-1.0f,  1.0f, -1.0f, 1.0f) + translate,
        scale * vec4(1.0f, -1.0f, -1.0f, 1.0f) + translate,
        scale * vec4(1.0f,  1.0f, -1.0f, 1.0f) + translate,
        scale * vec4(-1.0f, -1.0f,  1.0f, 1.0f) + translate,
        scale * vec4(-1.0f,  1.0f,  1.0f, 1.0f) + translate,
        scale * vec4(1.0f, -1.0f,  1.0f, 1.0f) + translate,
        scale * vec4(1.0f,  1.0f,  1.0f, 1.0f) + translate
    };
    vec4 n[6] =
    {
        vec4(0.0f, 0.0f, minus, 0.0f), vec4(0.0f, 0.0f, plus, 0.0f), vec4(minus, 0.0f, 0.0f, 0.0f),
        vec4(plus, 0.0f, 0.0f, 0.0f), vec4(0.0f, minus, 0.0f, 0.0f), vec4(0.0f, plus, 0.0f, 0.0f)
    };
    size_t indices[6][4] = { {0,1,2,3}, {4,5,6,7}, {0,1,4,5}, {2,3,6,7}, {0,2,4,6}, {1,3,5,7} };

    std::vector<vertex> vertexBuffer;
    for (size_t i = 0; i < 6; i++) {
        for (size_t j = 0; j < 4; j++) {
            vertexBuffer.push_back(vertex(v[indices[i][j]], n[i], colors[i], props));
        }
    }
    return vertexBuffer;
}

std::vector<uint32_t> createBoxIndexBuffer() {
    return std::vector<uint32_t>{
        0, 1, 2, 3, 1, 2,
        4, 5, 6, 7, 5, 6,
        8, 9, 11, 10, 11, 8,
        12, 13, 15, 14, 15, 12,
        16, 17, 19, 16, 18, 19,
        20, 21, 23, 20, 22, 23
    };
}

void createWorld(std::vector<primitive>& primitives, hitableContainer* container) {

    add(container,
        {
            sphere::create( vec4( 0.0f,  0.0f,  0.5f,  1.0f), 0.50f, vec4(0.80f, 0.30f, 0.30f, 1.00f), { 1.0f, 0.0f, 0.0f, pi, 0.0f}),
            sphere::create( vec4( 0.0f,  1.0f,  0.5f,  1.0f), 0.50f, vec4(0.80f, 0.80f, 0.80f, 1.00f), { 1.0f, 0.0f, 3.0f, 0.05f * pi, 0.0f }),
            sphere::create( vec4( 0.0f, -1.0f,  0.5f,  1.0f), 0.50f, vec4(0.90f, 0.90f, 0.90f, 1.00f), { 1.5f, 0.96f, 0.001f, 0.0f, 0.0f }),
            sphere::create( vec4( 0.0f, -1.0f,  0.5f,  1.0f), 0.45f, vec4(0.90f, 0.90f, 0.90f, 1.00f), { 1.0f / 1.5f, 0.96f, 0.001f, 0.0f, 0.0f }),
            sphere::create( vec4(-1.5f,  0.0f,  0.5f,  1.0f), 0.50f, vec4(1.00f, 0.90f, 0.70f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}),
            sphere::create( vec4( 1.5f, -1.5f,  0.2f,  1.0f), 0.20f, vec4(0.99f, 0.80f, 0.20f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}),
            sphere::create( vec4( 1.5f,  1.5f,  0.2f,  1.0f), 0.20f, vec4(0.20f, 0.80f, 0.99f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}),
            sphere::create( vec4(-1.5f, -1.5f,  0.2f,  1.0f), 0.20f, vec4(0.99f, 0.40f, 0.85f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}),
            sphere::create( vec4(-1.5f,  1.5f,  0.2f,  1.0f), 0.20f, vec4(0.40f, 0.99f, 0.50f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}),
            sphere::create( vec4(-0.5f, -0.5f,  0.2f,  1.0f), 0.20f, vec4(0.65f, 0.00f, 0.91f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}),
            sphere::create( vec4( 0.5f,  0.5f,  0.2f,  1.0f), 0.20f, vec4(0.80f, 0.70f, 0.99f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}),
            sphere::create( vec4(-0.5f,  0.5f,  0.2f,  1.0f), 0.20f, vec4(0.59f, 0.50f, 0.90f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}),
            sphere::create( vec4( 0.5f, -0.5f,  0.2f,  1.0f), 0.20f, vec4(0.90f, 0.99f, 0.50f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}),
            sphere::create( vec4(-1.0f, -1.0f,  0.2f,  1.0f), 0.20f, vec4(0.65f, 0.00f, 0.91f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}),
            sphere::create( vec4( 1.0f,  1.0f,  0.2f,  1.0f), 0.20f, vec4(0.80f, 0.90f, 0.90f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}),
            sphere::create( vec4(-1.0f,  1.0f,  0.2f,  1.0f), 0.20f, vec4(0.90f, 0.50f, 0.50f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}),
            sphere::create( vec4( 1.0f, -1.0f,  0.2f,  1.0f), 0.20f, vec4(0.50f, 0.59f, 0.90f, 1.00f), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f})
        }
    );

    primitives.emplace_back(
        createBoxVertexBuffer(vec4(3.0f, 3.0f, 1.5f, 1.0f), vec4(0.0f, 0.0f, 1.5f, 0.0f), sign::minus, { 1.0f, 0.0f, 0.0f, pi, 0.0f },
            { vec4(0.5f, 0.5f, 0.5f, 1.0f), vec4(0.5f, 0.5f, 0.5f, 1.0f), vec4(0.8f, 0.4f, 0.8f, 1.0f), vec4(0.4f, 0.4f, 0.4f, 1.0f), vec4(0.9f, 0.5f, 0.0f, 1.0f), vec4(0.1f, 0.4f, 0.9f, 1.0f) }),
        createBoxIndexBuffer(),
        0
    );
    primitives.back().moveToContainer(container);

    primitives.emplace_back(
        createBoxVertexBuffer(vec4(0.4f, 0.4f, 0.4f, 1.0f), vec4(1.5f, 0.0f, 0.4f, 0.0f), sign::plus, { 1.5f, 1.0f, 0.01f, 0.01f * pi, 0.0f },
            std::vector<vec4>(6, vec4(1.0f))),
        createBoxIndexBuffer(),
        0
    );
    primitives.back().moveToContainer(container);

    primitives.emplace_back(
        createBoxVertexBuffer(vec4(0.3f, 0.3f, 0.3f, 1.0f), vec4(1.5f, 0.0f, 0.4f, 0.0f), sign::plus, { 1.0f / 1.5f, 0.96f, 0.01f, 0.01f * pi, 0.0f },
            std::vector<vec4>(6, vec4(1.0f))),
        createBoxIndexBuffer(),
        0
    );
    primitives.back().moveToContainer(container);

    for (int i = 0; i < 0; i++) {
        float phi = 2.0f * pi * float(i) / 50.0f;
        primitives.emplace_back(
            createBoxVertexBuffer(vec4(0.1f, 0.1f, 0.1f, 1.0f), vec4(2.8f * std::cos(phi), 2.8f * std::sin(phi), 0.1f, 0.0f), sign::plus, { std::cos(phi), 0.96f, std::sin(phi), std::abs(std::sin(phi) * std::cos(phi)) * pi, 0.0f },
                std::vector<vec4>(6, vec4(std::abs(std::cos(phi)), std::abs(std::sin(phi)), std::abs(std::sin(phi) * std::cos(phi)), 1.0f))),
            createBoxIndexBuffer(),
            0
        );
        primitives.back().moveToContainer(container);
    }
}


