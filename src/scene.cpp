#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"

#define TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include "tinyobjloader/tiny_obj_loader.h"

using json = nlohmann::json;

Mesh::Mesh(){}

Mesh::~Mesh(){
    faces.clear();
}

const std::vector<Triangle> &Mesh::getFaces() {
    return faces;
}

/**
 * @brief This function loads an OBJ file and stores the vertices, normals, and UVs in the faces vector.
 * The code is taken from the tinyobjloader repository: https://github.com/tinyobjloader/tinyobjloader.
 * Although the original implementation can read meshes with arbitrarily-shaped faces, we are assuming that
 * the faces are triangles.
 * 
 * @param filepath The absolute path to the OBJ file.
 */
void Mesh::loadOBJ(const std::string &filepath) {
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "./"; // Path to material files

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filepath, reader_config)) {
        if (!reader.Error().empty()) {
            printf("TinyObjReader ERROR: %s\n", reader.Error().c_str());
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        printf("TinyObjReader WARNING: %s\n", reader.Warning().c_str());
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            std::vector<glm::vec3> verticesForOneFace;
            std::vector<glm::vec3> normalsForOneFace;
            std::vector<glm::vec2> uvsForOneFace;
            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3*size_t(idx.vertex_index)+0];
                tinyobj::real_t vy = attrib.vertices[3*size_t(idx.vertex_index)+1];
                tinyobj::real_t vz = attrib.vertices[3*size_t(idx.vertex_index)+2];
                verticesForOneFace.push_back(glm::vec3(vx, vy, vz));

                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3*size_t(idx.normal_index)+0];
                    tinyobj::real_t ny = attrib.normals[3*size_t(idx.normal_index)+1];
                    tinyobj::real_t nz = attrib.normals[3*size_t(idx.normal_index)+2];
                    normalsForOneFace.push_back(glm::vec3(nx, ny, nz));
                }

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2*size_t(idx.texcoord_index)+0];
                    tinyobj::real_t ty = attrib.texcoords[2*size_t(idx.texcoord_index)+1];
                    uvsForOneFace.push_back(glm::vec2(tx, ty));
                }

                // Optional: vertex colors
                // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
                // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
                // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
            }
            
            // We are assuming that each face is a triangle
            Triangle t(verticesForOneFace[0], verticesForOneFace[1], verticesForOneFace[2]);
            if (normalsForOneFace.size() > 0) {
                for (int i = 0; i < fv; i++) {
                    t.normals[i] = normalsForOneFace[i];
                }
            }
            if (uvsForOneFace.size() > 0) {
                for (int i = 0; i < fv; i++) {
                    t.uvs[i] = uvsForOneFace[i];
                }
            }
            this->faces.push_back(t);

            // per-face material
            shapes[s].mesh.material_ids[f];
            index_offset += fv;
        }
    }
}

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};

        if (p["TYPE"] == "Diffuse")
        {
            newMaterial.type = DIFFUSE;
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);  
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Mirror") {
            newMaterial.type = MIRROR;
            newMaterial.isSpecular = true;
        }
        else if (p["TYPE"] == "Dielectric") {
            newMaterial.type = DIELECTRIC;
            newMaterial.isSpecular = true;
        }
        else if (p["TYPE"] == "Microfacet") {
            newMaterial.type = MICROFACET;
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.roughness = p["ROUGHNESS"];
        }
        
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else if (type == "mesh")
        {
            newGeom.type = MESH;
            std::string filepath = p["MESH_PATH"];

            if (filepath.empty())
            {
                std::cerr << "No path provided for mesh object" << std::endl;
                exit(-1);
            }

            Mesh newMesh;
            newMesh.loadOBJ(filepath); // At this point the faces will be populated

            // Get the faces (triangles) from the Mesh object
            const std::vector<Triangle>& faces = newMesh.getFaces();
            size_t numTriangles = faces.size();

            if (numTriangles == 0)
            {
                std::cerr << "No triangles found in mesh object" << std::endl;
                exit(-1);
            }

            newGeom.triangles = new Triangle[numTriangles];

            // Copy the triangles from `Mesh` to `Geom`
            for (size_t i = 0; i < numTriangles; i++) {
                newGeom.triangles[i] = faces[i];
            }

            // Set the number of triangles (if needed for further use)
            newGeom.numTriangles = static_cast<int>(numTriangles);
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
