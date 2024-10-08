#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "scene.h"
#include <typeinfo>  // Required for typeid

Mesh::Mesh(){}

Mesh::~Mesh(){
    faces.clear();
    verts.clear();
    normals.clear();
    indices.clear();
    uvs.clear();

    if (albedoTexture) delete[] albedoTexture;
    if (normalTexture) delete[] normalTexture;
    if (bumpTexture) delete[] bumpTexture;
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
        loadOBJ(filepath, mesh.faces, mesh.verts, mesh.normals, mesh.indices, mesh.albedoTexture, mesh.normalTexture, mesh.bumpTexture); 
    } 
    else if (endsWith(filepath, ".gltf") || endsWith(filepath, ".glb")) {
        loadGLTFOrGLB(filepath, mesh.faces, mesh.verts, mesh.normals, mesh.indices, mesh.albedoTexture, mesh.normalTexture);
    }
    else {
        std::cerr << "Unsupported file format: " << filepath << std::endl;
        exit(-1);
    }
}

template <typename T>
void Scene::getValueFromJson(const json &data, const std::string &key, T &value) {
    // Check if the key exists in the JSON and if the value associated with it is not null
    if (data.contains(key) && !data[key].is_null()) {
        try {
            // Try to retrieve the value and assign it to the passed reference
            value = data.at(key).get<T>();
        } catch (const nlohmann::json::exception& e) {
            // Handle type mismatch or other errors
            // std::cerr << "Error getting value from JSON: " << e.what() << std::endl;
            printf("Error getting value from JSON: %s\n", e.what());
        }
    }
    else {
        printf("Key %s not found in JSON\n", key.c_str());
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);

    // Reading materials
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;

    for (const auto& item : materialsData.items())
    {   
        // Here name must be unique for each material
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

    // Reading textures
    const auto& texturesData = data["Textures"];
    std::unordered_map<std::string, uint32_t> AlbedoTexToID;
    std::unordered_map<std::string, uint32_t> NormalTexToID;
    std::unordered_map<std::string, uint32_t> BumpTexToID;

    for (const auto& texture : texturesData.items()) {
        // Here name must be unique for each texture
        const auto& name = texture.key();
        const auto& p = texture.value();

        std::string textureType = p["TYPE"];
        if (textureType.empty())
        {
            std::cerr << "You specify a texture but you haven't specify the type of it." << std::endl;
            continue;
        }
        else if (textureType != "Albedo" && textureType != "Normal" && textureType != "Bump") {
            std::cerr << "Unsupported texture type: " << textureType << std::endl;
            continue;
        }

        std::string filepath;
        getValueFromJson(p, "TEXTURE_PATH", filepath);

        if (filepath.empty())
        {
            std::cerr << "No path provided for the texture. Cannot load." << std::endl;
            continue;
        }

        glm::vec4* curTexture;
        glm::ivec2 textureSize;
        loadTexture(filepath, textureType, curTexture, textureSize);

        if (p["TYPE"] == "Albedo") {
            AlbedoTexToID[name] = albedoTextures.size();
            albedoTextures.emplace_back(make_tuple(curTexture, textureSize));
            printf("Albedo texture added with ID: %d\n", AlbedoTexToID[name]);
        }
        else if (p["TYPE"] == "Normal") {
            NormalTexToID[name] = normalTextures.size();
            normalTextures.emplace_back(make_tuple(curTexture, textureSize));
            printf("Normal texture added with ID: %d\n", NormalTexToID[name]);
        }
        else if (p["TYPE"] == "Bump") {
            BumpTexToID[name] = bumpTextures.size();
            bumpTextures.emplace_back(make_tuple(curTexture, textureSize));
            printf("Bump texture added with ID: %d\n", BumpTexToID[name]);
        }
    }
    
    // Reading objects
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        const std::string& mat = p["MATERIAL"];

        Geom newGeom;

        // Have to initialize the material IDs to -1, otherwise the default value for int is 0 
        // N we will have segmentation fault in CUDA
        newGeom.material.albedoTextureID = -1;
        newGeom.material.normalTextureID = -1;
        newGeom.material.bumpTextureID = -1;

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

            std::string filepath;
            getValueFromJson(p, "MESH_PATH", filepath);

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

            for (size_t i = 0; i < numTriangles; i++) {
                // Copy the triangles from `Mesh` to `Geom`
                newGeom.triangles[i] = triangles[i];
            }

            /** Here we are populating triangles from BVH **/
            BVH bvh = BVH(newMesh.verts.data(), newMesh.normals.data(), newMesh.indices.data(), newMesh.indices.size());
            newGeom.bvhTriangles = bvh.allTriangles;
            newGeom.bvhNodes = bvh.allNodes.nodes;
            newGeom.numBvhNodes = bvh.allNodes.nodeCount(); 
            
            /** Here we are reading the textures if there are any **/
            std::vector<std::string> textures;
            getValueFromJson(p, "TEXTURES", textures);

            if (textures.size() == 0) {
                std::cerr << "This mesh is not using any textures." << std::endl;
            }
            else {
                for (const auto& texture : textures) {
                    std::string textureName = texture;
                    bool findAlbedo = AlbedoTexToID.find(textureName) != AlbedoTexToID.end();
                    bool findNormal = NormalTexToID.find(textureName) != NormalTexToID.end();
                    bool findBump = BumpTexToID.find(textureName) != BumpTexToID.end();

                    if (!findAlbedo && !findNormal && !findBump) {
                        std::cerr << "Texture " << textureName << " not found in the scene" << std::endl;
                    }

                    newGeom.material.albedoTextureID = findAlbedo ? AlbedoTexToID[textureName] : -1;
                    newGeom.material.normalTextureID = findNormal ? NormalTexToID[textureName] : -1;
                    newGeom.material.bumpTextureID = findBump ? BumpTexToID[textureName] : -1;
                }
            }
        }
        
        newGeom.material.materialid = MatNameToID[mat];

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
    // Use getValueFromJson if it's a custom type
    getValueFromJson(cameraData, "LENS_RADIUS", camera.lensRadius);
    getValueFromJson(cameraData, "FOCAL_DISTANCE", camera.focalDistance);

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

    printf("Scene loaded with %d materials, %d albedo textures, %d normal textures, %d bump textures, %d objects, %d lights\n",
        materials.size(), albedoTextures.size(), normalTextures.size(), bumpTextures.size(), geoms.size(), lights.size());
}
