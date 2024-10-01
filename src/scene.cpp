#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
#include "tiny_gltf.h"
#include "tiny_obj_loader.h"

using json = nlohmann::json;


Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
#if 0
        for (int i = 0; i < geoms.size(); ++i)
        {
            cout << "Geom " << i << endl;
            cout << "Type: " << geoms[i].type << endl;
            //suppose to be 1
            cout << "HasTexture: " << geoms[i].hasTexture << endl;
            //suppose to be 0
            cout << "Texture ID: " << geoms[i].textureid << endl;
            cout << "Material ID: " << geoms[i].materialid << endl;
            cout << "Translation: " << glm::to_string(geoms[i].translation) << endl;
            cout << "Rotation: " << glm::to_string(geoms[i].rotation) << endl;
            cout << "Scale: " << glm::to_string(geoms[i].scale) << endl;
            cout << "Transform: " << glm::to_string(geoms[i].transform) << endl;
            cout << "Inverse Transform: " << glm::to_string(geoms[i].inverseTransform) << endl;
            cout << "InvTranspose: " << glm::to_string(geoms[i].invTranspose) << endl;
            cout << " " << endl;
        }
#endif
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
//---------------------Assign materials from JSON---------------------
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            const auto& specCol = p["SPECRGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = glm::vec3(specCol[0], specCol[1], specCol[2]);
            newMaterial.indexOfRefraction = p["IOR"];
            newMaterial.hasRefractive = 1.0f;
        }
        else if (p["TYPE"] == "Glass")
        {
            const auto& col = p["RGB"];
            const auto& specCol = p["SPECRGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = glm::vec3(specCol[0], specCol[1], specCol[2]);
            newMaterial.indexOfRefraction = p["IOR"];
            newMaterial.hasRefractive = 1.0f;
            newMaterial.hasReflective = 1.0f;
        }
        else if (p["TYPE"] == "Reflective")
        {
            const auto& col = p["RGB"];
            //newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.exponent = p["EXPONENT"];
            newMaterial.hasReflective = 1.0f;
        }

        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);

    }
//---------------------Assign camera and objects from JSON---------------------
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
            newGeom.materialid = MatNameToID[p["MATERIAL"]];
            std::cout << "CUBE MATERIALID is:" << newGeom.materialid << endl;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
            newGeom.materialid = MatNameToID[p["MATERIAL"]];
            std::cout << "SPHERE MATERIALID is:" << newGeom.materialid << endl;
        }
        else if (type == "mesh")
        {
            newGeom.type = MESH;
#if 1
            //Loading vertices, normals, uvs
            //Read mtl file
            loadFromOBJ(p["OBJ"], newGeom, MatNameToID);
            std::cout << "Loaded mesh from " << p["OBJ"] << endl;
            //Add for texture
            if (p.contains("TEXTURE")) {
                newGeom.hasTexture = 1;
                std::cout << "Loaded texture from " << p["TEXTURE"] << endl;
                loadTexture(p["TEXTURE"], newGeom, "../scenes/");
                std::cout << "texture id is " << newGeom.textureid << endl;
            }
            else {
                std::cout << "No [TEXUTRE] section found" << endl;
            }
            std::cout << "MESH MATERIALID is:" << newGeom.materialid << endl;
#endif
        }
        //newGeom.materialid = MatNameToID[p["MATERIAL"]];
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
    //print out all the materials
    for (int i = 0; i < materials.size(); ++i) {
        cout << "Material " << i << endl;
		cout << "Color: " << glm::to_string(materials[i].color) << endl;
		cout << "Emittance: " << materials[i].emittance << endl;
		cout << "IndexOfRefraction: " << materials[i].indexOfRefraction << endl;
		cout << "HasRefractive: " << materials[i].hasRefractive << endl;
		cout << "Specular Color: " << glm::to_string(materials[i].specular.color) << endl;
		cout << "Specular Exponent: " << materials[i].specular.exponent << endl;
		cout << "HasReflective: " << materials[i].hasReflective << endl;
		cout << " " << endl;
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

void Scene::loadTexture(const std::string& filename, Geom& newGeom, std::string path) {
    int width, height, channels;
    unsigned char* data = stbi_load((path + filename).c_str(), &width, &height, &channels, 0);
    if (!data) {
        std::cerr << "Failed to load texture: " << path + filename << std::endl;
        exit(1);
    }

    // Create a new texture
    Texture newTexture;
    newTexture.width = width;
    newTexture.height = height;
    newTexture.channels = channels;
    newTexture.data = data;

    // Add the texture to the scene
    textures.push_back(newTexture);
    newGeom.textureid = textures.size() - 1;
}

// Reference to the tinyobj loader example:
// https://github.com/tinyobjloader/tinyobjloader/blob/release/examples/viewer/viewer.cc

void Scene::loadFromOBJ(const std::string& filename, Geom& newGeom, std::unordered_map<std::string, uint32_t>& MatNameToID) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> tobj_materials;
    std::string warn, err;
    const std::string path = "../scenes/";
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &tobj_materials, &warn, &err, (path + filename).c_str(), path.c_str());
    cout<< "Loading from OBJ: " << filename << endl;

    if (!warn.empty()) {
        std::cout << "WARNING: " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << "ERROR: " << err << std::endl;
    }
    if (!ret) {
        std::cerr << "Failed to load " << filename << std::endl;
        exit(1);
    }
#if 0
    // Print info
    printf("# of vertices  = %d\n", (int)(attrib.vertices.size()) / 3);
    printf("# of normals   = %d\n", (int)(attrib.normals.size()) / 3);
    printf("# of texcoords = %d\n", (int)(attrib.texcoords.size()) / 2);
    printf("# of materials = %d\n", (int)tobj_materials.size());
    printf("# of shapes    = %d\n", (int)shapes.size());
#endif
#if 1
    for (size_t matID = 0; matID < tobj_materials.size(); matID++) {
        const tinyobj::material_t& mat = tobj_materials[matID];
        // Print material name
        printf("material[%d].name = %s\n", int(matID), mat.name.c_str());

        // Print ambient color
        printf("material[%d].ambient = (%f, %f, %f)\n", int(matID),
            mat.ambient[0], mat.ambient[1], mat.ambient[2]);

        // Print diffuse color
        printf("material[%d].diffuse = (%f, %f, %f)\n", int(matID),
            mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);

        // Print specular color
        printf("material[%d].specular = (%f, %f, %f)\n", int(matID),
            mat.specular[0], mat.specular[1], mat.specular[2]);

        Material geoMat{};

        /***********************************************
         * Mapping tinyobj::material_t to Material struct
         *
         * tinyobj::material_t        |    Material
         * ------------------------------------------------
         * mat.diffuse                | geoMat.color
         * - Diffuse color            | - Base color
         * ------------------------------------------------
         * mat.emission               | geoMat.emittance
         * - Emissive color           | - Light emission
         * ------------------------------------------------
         * mat.transmittance          | geoMat.hasRefractive
         * - Transparency             | - Refractive flag
         * ------------------------------------------------
         * mat.ior                    | geoMat.indexOfRefraction
         * - Index of refraction      | - Refraction index
         * ------------------------------------------------
         * mat.specular               | geoMat.specular.color
         * - Specular color           | - Highlight color
         * ------------------------------------------------
         * mat.shininess              | geoMat.specular.exponent
         * - Shininess factor         | - Specular exponent
         * ------------------------------------------------
         * mat.illum                  | Illumination model
         * - Illumination mode        | - Reflect/refract flag
         ***********************************************/


        // Set the diffuse color (Kd) as the base color
        geoMat.color = glm::vec3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);

        // Handle emittance like light sources
        glm::vec3 emissive = glm::vec3(mat.emission[0], mat.emission[1], mat.emission[2]);
        geoMat.emittance = glm::length(emissive) > 0.0f ? glm::length(emissive) : 0.0f;

        // Handle transparency and refraction
        glm::vec3 transparency = glm::vec3(mat.transmittance[0], mat.transmittance[1], mat.transmittance[2]);
        if (glm::length(transparency) > 0.0f) {
            geoMat.hasRefractive = 1.0f;
            geoMat.indexOfRefraction = mat.ior; 
        }
        else {
            geoMat.hasRefractive = 0.0f;
            geoMat.indexOfRefraction = 1.0f;
        }

        if (mat.illum == 1) {
            // No specular reflection, only diffuse
            geoMat.hasReflective = 0.0f;
        }
        else if (mat.illum == 2) {
            // Diffuse and specular reflection
            geoMat.specular.color = glm::vec3(mat.specular[0], mat.specular[1], mat.specular[2]);
            geoMat.specular.exponent = mat.shininess;
            geoMat.hasReflective = mat.shininess > 0.0f ? 1.0f : 0.0f;
        }
        else if (mat.illum == 3 || mat.illum == 4) {
            // Transparency and reflection (diffuse + transparency)
            geoMat.specular.color = glm::vec3(mat.specular[0], mat.specular[1], mat.specular[2]);
            geoMat.specular.exponent = mat.shininess;
            geoMat.hasReflective = 1.0f;  // Reflective
            geoMat.hasRefractive = 1.0f;  // Refractive
        }
        else if (mat.illum == 5 || mat.illum == 6) {
            geoMat.specular.color = glm::vec3(mat.specular[0], mat.specular[1], mat.specular[2]);
            geoMat.specular.exponent = mat.shininess;
            geoMat.hasReflective = 1.0f;
        }
        else {
            // Default case for other illumination models
            geoMat.specular.color = glm::vec3(mat.specular[0], mat.specular[1], mat.specular[2]);
            geoMat.specular.exponent = mat.shininess;
            geoMat.hasReflective = mat.shininess > 0.0f ? 1.0f : 0.0f;
        }

        materials.emplace_back(geoMat);
        MatNameToID[mat.name] = materials.size() - 1;
        if (!mat.diffuse_texname.empty()) {
            std::cout << "Loading texture in loadFromOBJ!: " << mat.diffuse_texname << std::endl;
			loadTexture(mat.diffuse_texname, newGeom, path);
			newGeom.hasTexture = 1;
        }
       // newGeom.materialid = materials.size() - 1;
    }

#endif

    // Start of triangle indices for this geometry
    newGeom.triIndexStart = triangles.size();

    for (const auto& shape : shapes) {
        //int numTrianglesInShape = shape.mesh.indices.size() / 3;
        //std::cout << "Triangles in shape: " << numTrianglesInShape << std::endl;
        for (size_t i = 0; i < shape.mesh.indices.size(); i += 3) {
            int idx0 = shape.mesh.indices[i].vertex_index;
            int idx1 = shape.mesh.indices[i + 1].vertex_index;
            int idx2 = shape.mesh.indices[i + 2].vertex_index;

            // Vertices
            glm::vec3 v0(attrib.vertices[3 * idx0], attrib.vertices[3 * idx0 + 1], attrib.vertices[3 * idx0 + 2]);
            glm::vec3 v1(attrib.vertices[3 * idx1], attrib.vertices[3 * idx1 + 1], attrib.vertices[3 * idx1 + 2]);
            glm::vec3 v2(attrib.vertices[3 * idx2], attrib.vertices[3 * idx2 + 1], attrib.vertices[3 * idx2 + 2]);
            //std::cout << "V0: " << v0.x << ", " << v0.y << ", " << v0.z << std::endl;
            //std::cout << "V1: " << v1.x << ", " << v1.y << ", " << v1.z << std::endl;
            //std::cout << "V2: " << v2.x << ", " << v2.y << ", " << v2.z << std::endl;

            // UVs
            glm::vec2 uv0(0.0f), uv1(0.0f), uv2(0.0f);
            if (!attrib.texcoords.empty()) {
                int texIdx0 = shape.mesh.indices[i].texcoord_index;
                int texIdx1 = shape.mesh.indices[i + 1].texcoord_index;
                int texIdx2 = shape.mesh.indices[i + 2].texcoord_index;

                uv0 = glm::vec2(attrib.texcoords[2 * texIdx0], attrib.texcoords[2 * texIdx0 + 1]);
                uv1 = glm::vec2(attrib.texcoords[2 * texIdx1], attrib.texcoords[2 * texIdx1 + 1]);
                uv2 = glm::vec2(attrib.texcoords[2 * texIdx2], attrib.texcoords[2 * texIdx2 + 1]);
                //std::cout << "UV0: " << uv0.x << ", " << uv0.y << std::endl;
                //std::cout << "UV1: " << uv1.x << ", " << uv1.y << std::endl;
                //std::cout << "UV2: " << uv2.x << ", " << uv2.y << std::endl;
            }

            // Normals
            glm::vec3 n0(0.0f), n1(0.0f), n2(0.0f);
            if (!attrib.normals.empty()) {
                int normIdx0 = shape.mesh.indices[i].normal_index;
                int normIdx1 = shape.mesh.indices[i + 1].normal_index;
                int normIdx2 = shape.mesh.indices[i + 2].normal_index;

                n0 = glm::vec3(attrib.normals[3 * normIdx0], attrib.normals[3 * normIdx0 + 1], attrib.normals[3 * normIdx0 + 2]);
                n1 = glm::vec3(attrib.normals[3 * normIdx1], attrib.normals[3 * normIdx1 + 1], attrib.normals[3 * normIdx1 + 2]);
                n2 = glm::vec3(attrib.normals[3 * normIdx2], attrib.normals[3 * normIdx2 + 1], attrib.normals[3 * normIdx2 + 2]);
            }

            // Tangents and Bitangents
            glm::vec3 edge1 = v1 - v0;
            glm::vec3 edge2 = v2 - v0;
            glm::vec2 deltaUV1 = uv1 - uv0;
            glm::vec2 deltaUV2 = uv2 - uv0;
            float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
            glm::vec3 tangent;
            tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
            tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
            tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
            tangent = glm::normalize(tangent);
            glm::vec3 bitangent;
            bitangent.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
            bitangent.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
            bitangent.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);
            bitangent = glm::normalize(bitangent);

            // Push the triangle into the Scene's triangle vector
            //triangles.push_back({ v0, v1, v2, uv0, uv1, uv2, n0, n1, n2 });
            triangles.push_back({
            {v0, v1, v2},  
            {uv0, uv1, uv2},  
            {n0, n1, n2},  
            tangent,  
            bitangent 
             });
        }
    }

    // End of triangle indices for this geometry
    newGeom.triIndexEnd = triangles.size();
    //currently one mesh has one material
    int materialID = shapes[0].mesh.material_ids[0];
    newGeom.materialid = MatNameToID[tobj_materials[materialID].name];

    //std::cout << "Vertices: " << attrib.vertices.size() / 3 << std::endl;
    //std::cout << "Triangles Loaded " << newGeom.triIndexEnd - newGeom.triIndexStart << " triangles for geometry.\n";
}