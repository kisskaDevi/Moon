#ifndef OBJMODEL_H
#define OBJMODEL_H
#include "model.h"

#include <filesystem>
#include <vector>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

namespace cuda::rayTracing {

struct ObjModelInfo{
    Properties props{};
    vec4f color{0.0f};
    bool mergeVertices{false};
    bool useModelColors{false};

    ObjModelInfo(
        Properties props = {},
        vec4f color = {0.0f},
        bool mergeVertices = false,
        bool useModelColors = false) :
        props(props), color(color), mergeVertices(mergeVertices), useModelColors(useModelColors)
    {}
};

class ObjModel : public Model {
private:
    std::filesystem::path path;
    ObjModelInfo info;

public:
    ObjModel(const std::filesystem::path& path, const ObjModelInfo& info = {}) :
        path(path),
        info(info)
    {}

    ObjModel(const std::vector<Vertex>& vertexBuffer, const std::vector<uint32_t>& indexBuffer) : Model(vertexBuffer, indexBuffer) {}

    void load(const mat4f& transform) override {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;

        if (std::string warn, err; !tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str()))
            throw std::runtime_error(warn + err);

        std::unordered_map<std::string, uint32_t> verticesMap;
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;

        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                Vertex vertex{};

                if(attrib.vertices.size() > 0){
                    vertex.point = {
                        attrib.vertices[3 * index.vertex_index + 0],
                        attrib.vertices[3 * index.vertex_index + 1],
                        attrib.vertices[3 * index.vertex_index + 2],
                        1.0f
                    };
                    vertex.point = transform * vertex.point;
                }

                if (info.useModelColors && attrib.colors.size() > 0) {
                    vertex.color += {
                        attrib.colors[3 * index.vertex_index + 0],
                        attrib.colors[3 * index.vertex_index + 1],
                        attrib.colors[3 * index.vertex_index + 2],
                        1.0f
                    };
                } else {
                    vertex.color = info.color;
                }

                if(attrib.normals.size() > 0){
                    vertex.normal = {
                        attrib.normals[3 * index.normal_index + 0],
                        attrib.normals[3 * index.normal_index + 1],
                        attrib.normals[3 * index.normal_index + 2],
                        0.0f
                    };
                    vertex.normal = transform * vertex.normal;
                }

                vertex.props = info.props;

                uint32_t newindex = indices.size();
                if(info.mergeVertices){
                    std::string id = std::to_string(vertex.point.x()) + "_" + std::to_string(vertex.point.y()) + "_" + std::to_string(vertex.point.z());
                    if(verticesMap.count(id) > 0){
                        newindex = verticesMap[id];
                    } else {
                        verticesMap[id] = newindex;
                    }
                }

                indices.push_back(newindex);
                vertices.push_back(vertex);
            }
        }

        if(attrib.normals.size() == 0){
            for(uint32_t i = 0; i < indices.size(); i += 3){
                const vec4f n = normal(cross(
                    vertices[indices[i + 1]].point - vertices[indices[i + 0]].point,
                    vertices[indices[i + 2]].point - vertices[indices[i + 1]].point
                ));

                vertices[indices[i + 0]].normal += n;
                vertices[indices[i + 1]].normal += n;
                vertices[indices[i + 2]].normal += n;
            }
            for(uint32_t i = 0; i < vertices.size(); i++){
                vertices[i].normal = normal(vertices[i].normal);
            }
        }

        *this = ObjModel(vertices, indices);
    }
};

}
#endif // OBJMODEL_H
