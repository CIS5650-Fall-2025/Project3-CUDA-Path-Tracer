#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "scene.h"

Mesh::Mesh(){}

Mesh::~Mesh(){
    faces.clear();
    m_cdf.clear();
}

const float Mesh::computeTriangleArea(const Triangle &t) {
    glm::vec3 e1 = t.points[1] - t.points[0];
    glm::vec3 e2 = t.points[2] - t.points[0];

    return 0.5f * glm::length(glm::cross(e1, e2));
}

void Mesh::addTriangleAreaToCDF(const float area) {
    float lastCDF = m_cdf.size() > 0 ? m_cdf[m_cdf.size() - 1] : 0.0f;
    m_cdf.push_back(lastCDF + area);
}

void Mesh::normaliseCDF() {
    float totalArea = m_cdf[m_cdf.size() - 1];

    if (totalArea == 0.0f) {
        std::cerr << "Can't normalise CDF because the total area of this mesh is zero." << std::endl;
        exit(-1);
    }

    for (size_t i = 0; i < m_cdf.size(); i++) {
        m_cdf[i] /= totalArea;
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

void Scene::loadMesh(const std::string &filepath, Mesh &mesh) {
    if (endsWith(filepath, ".obj")) {
        printf("Loading OBJ file: %s\n", filepath.c_str());
        loadOBJ(filepath, mesh.faces); 
    } 
    else if (endsWith(filepath, ".gltf") || endsWith(filepath, ".glb")) {
        loadGLTFOrGLB(filepath, mesh.faces);
    }
    else {
        std::cerr << "Unsupported file format: " << filepath << std::endl;
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
            const auto& spec_col = p["SPEC_RGB"];
            newMaterial.specularColor = glm::vec3(spec_col[0], spec_col[1], spec_col[2]);
        }
        else if (p["TYPE"] == "Dielectric") {
            newMaterial.type = DIELECTRIC;
            const auto& spec_col = p["SPEC_RGB"];
            newMaterial.specularColor = glm::vec3(spec_col[0], spec_col[1], spec_col[2]);
            newMaterial.indexOfRefraction = p["IOR"];
        }
        else if (p["TYPE"] == "Microfacet") {
            newMaterial.type = MICROFACET;
            const auto& col = p["RGB"];
            const auto& spec_col = p["SPEC_RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specularColor = glm::vec3(spec_col[0], spec_col[1], spec_col[2]);
            newMaterial.roughness = p["ROUGHNESS"];
            newMaterial.indexOfRefraction = p["IOR"];
        }
        
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        const std::string& mat = p["MATERIAL"];

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
            loadMesh(filepath, newMesh);

            // Get the faces (triangles) from the Mesh object
            std::vector<Triangle>& triangles = newMesh.faces;
            size_t numTriangles = triangles.size();

            if (numTriangles == 0)
            {
                std::cerr << "No triangles found in mesh object" << std::endl;
                exit(-1);
            }

            newGeom.numTriangles = static_cast<int>(numTriangles);
            newGeom.triangles = new Triangle[numTriangles];

            /** All this area and CDF stuff is for potential light sampling **/
            float surfaceArea = 0.0f;

            for (size_t i = 0; i < numTriangles; i++) {
                const float area = newMesh.computeTriangleArea(triangles[i]);
                surfaceArea += area;
                newMesh.addTriangleAreaToCDF(area);
            }

            // Normalise CDF once all triangles have been added
            newMesh.normaliseCDF();

            for (size_t i = 0; i < numTriangles; i++) {
                triangles[i].cdf = newMesh.m_cdf[i];
                // Copy the triangles from `Mesh` to `Geom`
                newGeom.triangles[i] = triangles[i];
                // printf("Triangle %zu CDF: %f\n", i, newGeom.triangles[i].cdf);
            }

            newGeom.area = surfaceArea;

            /** Here we are populating triangles from BVH **/
            BVH bvh = BVH(triangles);

            // newGeom.bvhTriangles = bvh.allTris;
            exit(-1);
        }

        newGeom.materialid = MatNameToID[mat];
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
        if (mat == "light")
        {
            lights.push_back(newGeom);
        }
    }

    if (lights.size() == 0)
    {
        std::cerr << "No lights found in the scene, your render will be pitch black!" << std::endl;
        exit(-1);
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
