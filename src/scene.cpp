#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

using json = nlohmann::json;

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

//void loadMesh()
//{
//    tinygltf::Model model;
//    tinygltf::TinyGLTF gltf_ctx;
//    std::string err;
//    std::string warn;
//    std::string input_filename{ "mesh.glb" };
//    std::string ext{ "glb" };
//
//    bool ret = false;
//    if (ext.compare("glb") == 0) {
//        std::cout << "Reading binary glTF" << std::endl;
//        // assume binary glTF.
//        ret = gltf_ctx.LoadBinaryFromFile(&model, &err, &warn,
//            input_filename.c_str());
//    }
//    else {
//        std::cout << "Reading ASCII glTF" << std::endl;
//        // assume ascii glTF.
//        ret =
//            gltf_ctx.LoadASCIIFromFile(&model, &err, &warn, input_filename.c_str());
//    }
//
//    if (!warn.empty()) {
//        printf("Warn: %s\n", warn.c_str());
//    }
//
//    if (!err.empty()) {
//        printf("Err: %s\n", err.c_str());
//    }
//
//    if (!ret) {
//        printf("Failed to parse glTF\n");
//        return -1;
//    }
//
//}

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
            newMaterial.hasReflective = 1.0f;
        }
        else if (p["TYPE"] == "Transparent")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = 1.55;
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];

        const auto& transJson = p["TRANS"];
        const auto& rotatJson = p["ROTAT"];
        const auto& scaleJson = p["SCALE"];
		const std::string material = p["MATERIAL"];

        glm::vec3 translation = glm::vec3(transJson[0], transJson[1], transJson[2]);
		glm::vec3 rotation = glm::vec3(rotatJson[0], rotatJson[1], rotatJson[2]);
		glm::vec3 scale = glm::vec3(scaleJson[0], scaleJson[1], scaleJson[2]);

        if (type == "mesh") {
            int materialID;
            if (material == "GLTF") {
                materialID = -1;
            }
			else {
				materialID = MatNameToID[material];
			}
            loadGLTF(p["FILE"], materialID, translation, rotation, scale);
            continue;
        }

        Geom newGeom;
        if (type == "cube")  newGeom.type = CUBE;
        else if (type == "sphere") newGeom.type = SPHERE;
		else
        {
			std::cerr << "Unknown object type: " << type << std::endl;
        }

        newGeom.materialid = MatNameToID[material];
        newGeom.translation = translation;
        newGeom.rotation = rotation;
        newGeom.scale = scale;
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
    camera.aperture = cameraData["APERTURE"];

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


std::string getFileExtension(const std::string& filename)
{
    return filename.substr(filename.find_last_of(".") + 1);
}

template <typename T, typename U>
std::vector<U> castBufferToVector(const unsigned char* buffer, size_t offset, size_t count, int stride)
{
    std::vector<U> result;
    result.reserve(count);

    for (size_t i = 0; i < count; ++i)
    {
        T value;
        std::memcpy(&value, &buffer[offset + i * stride], sizeof(T));
        result.push_back(static_cast<U>(value));
    }
    return result;
}

///
/// Loads glTF 2.0 mesh
///
// bool LoadGLTF(const std::string &filename, float scale,
//               std::vector<Mesh<float>> *meshes,
//               std::vector<Material> *materials,
//               std::vector<Texture> *textures)
bool Scene::loadGLTF(const std::string& filename, int materialID, const glm::vec3 translation, const glm::vec3 rotation, const glm::vec3 scale)
//std::vector<Mesh<float>>* meshes)
{
    //

    tinygltf::Model model;
    tinygltf::TinyGLTF loader;

    const std::string fileExtension = getFileExtension(filename);

    bool successfulLoad = false;
    std::string err;
    std::string warn;
    if (fileExtension == "glb")
    {
        // Binary input
        successfulLoad = loader.LoadBinaryFromFile(&model, &err, &warn, filename.c_str());
    }
    else if (fileExtension == "gltf")
    {
        // ASCII input
        successfulLoad = loader.LoadASCIIFromFile(&model, &err, &warn, filename.c_str());
    }

    // Error handling
    if (!warn.empty())
        std::cout << "glTF parse warning: " << warn << std::endl;
    if (!err.empty())
        std::cerr << "glTF parse error: " << err << std::endl;
    if (!successfulLoad)
    {
        std::cerr << "Failed to load glTF: " << filename << std::endl;
        return false;
    }
    //std::vector<std::vector<Triangle>> meshTriangleVector;
    //std::vector<std::vector<glm::vec3>> meshVertexVector;
    int triangleIndex = 0;
    int startIndex = 0;
    // Iterate through all the meshes in the glTF file
    for (const auto& gltfMesh : model.meshes)
    {
        // Create a mesh object
        //tinygltf::Mesh<float> loadedMesh(sizeof(float) * 3);

        // To store the min and max of the buffer (as 3D vector of floats)
        // glm::vec3 bbMin, bbMax;

        std::vector<TriangleIdx> triangleIndices;
        std::vector<glm::vec3> triangleVertices;
        //std::vector<glm::vec3> vertexVector;
        //std::vector<int> triangleIndices;

        // Store the name of the glTF mesh (if defined)
        // loadedMesh.name = gltfMesh.name;

        // For each primitive
        for (const auto& meshPrimitive : gltfMesh.primitives)
        {
            // Boolean used to check if we have converted the vertex buffer format
            bool convertedToTriangleList = false;

            //std::vector<int> indicesVector;

            const auto& indicesAccessor = model.accessors[meshPrimitive.indices];
            const auto& bufferView = model.bufferViews[indicesAccessor.bufferView];
            const auto& buffer = model.buffers[bufferView.buffer];
            // Calculate starting adress of data
            // const auto dataAddress = buffer.data.data() + bufferView.byteOffset +
                                        // indicesAccessor.byteOffset;

            const size_t byteOffset = bufferView.byteOffset + indicesAccessor.byteOffset;
            const int byteStride = indicesAccessor.ByteStride(bufferView);
            const size_t count = indicesAccessor.count;

            std::vector<int> indicesVector;
            switch (indicesAccessor.componentType)
            {
            case TINYGLTF_COMPONENT_TYPE_BYTE:
                indicesVector = castBufferToVector<char, int>(
                    buffer.data.data(), byteOffset, count, byteStride);
                break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                indicesVector = castBufferToVector<unsigned short, int>(
                    buffer.data.data(), byteOffset, count, byteStride);
                break;
            case TINYGLTF_COMPONENT_TYPE_SHORT:
                indicesVector = castBufferToVector<short, int>(
                    buffer.data.data(), byteOffset, count, byteStride);
                break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                indicesVector = castBufferToVector<unsigned short, int>(
                    buffer.data.data(), byteOffset, count, byteStride);
                break;
            case TINYGLTF_COMPONENT_TYPE_INT:
                indicesVector = castBufferToVector<int, int>(
                    buffer.data.data(), byteOffset, count, byteStride);
                break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                indicesVector = castBufferToVector<unsigned int, int>(
                    buffer.data.data(), byteOffset, count, byteStride);
                break;
            default:
                break;
            }



            //for (size_t i = 0; i < indicesVector.size(); ++i)
            //{
            //    loadedMesh.faces.push_back(indicesVector[i]);
            //}

            switch (meshPrimitive.mode)
            {
                // We re-arrange the indices so that it describe a simple list of
                // triangles
            case TINYGLTF_MODE_TRIANGLE_FAN:
                if (!convertedToTriangleList)
                {
                    // This only has to be done once per primitive
                    convertedToTriangleList = true;

                    // We steal the guts of the vector
                    //auto triangleFan = std::move(loadedMesh.faces);
                    //loadedMesh.faces.clear();

                    // Push back the indices that describe just one triangle one by one
                    for (size_t i = 2; i < indicesVector.size(); ++i)
                    {
                        triangleIndices.push_back(TriangleIdx{ indicesVector[0], indicesVector[i - 1], indicesVector[i] });
                        /*loadedMesh.faces.push_back(triangleFan[0]);
                        loadedMesh.faces.push_back(triangleFan[i - 1]);
                        loadedMesh.faces.push_back(triangleFan[i]);*/
                    }
                }
            case TINYGLTF_MODE_TRIANGLE_STRIP:
                if (!convertedToTriangleList)
                {
                    // This only has to be done once per primitive
                    convertedToTriangleList = true;

                    /*     auto triangleStrip = std::move(loadedMesh.faces);
                         loadedMesh.faces.clear();*/

                    for (size_t i = 2; i < indicesVector.size(); ++i)
                    {
                        triangleIndices.push_back(TriangleIdx{ indicesVector[i - 2], indicesVector[i - 1], indicesVector[i] });
                        //loadedMesh.faces.push_back(triangleStrip[i - 2]);
                        //loadedMesh.faces.push_back(triangleStrip[i - 1]);
                        //loadedMesh.faces.push_back(triangleStrip[i]);
                    }
                }
            case TINYGLTF_MODE_TRIANGLES: // this is the simpliest case to handle

            {
                for (size_t i = 2; i < indicesVector.size(); i += 3)
                {
                    triangleIndices.push_back(TriangleIdx{ indicesVector[i - 2], indicesVector[i - 1], indicesVector[i] });
                    //loadedMesh.faces.push_back(triangleStrip[i - 2]);
                    //loadedMesh.faces.push_back(triangleStrip[i - 1]);
                    //loadedMesh.faces.push_back(triangleStrip[i]);
                }

                for (const auto& attribute : meshPrimitive.attributes)
                {
                    const auto attribAccessor = model.accessors[attribute.second];
                    const auto& bufferView = model.bufferViews[attribAccessor.bufferView];
                    const auto& buffer = model.buffers[bufferView.buffer];

                    //const int byteStride = attribAccessor.ByteStride(bufferView);
                    //const auto dataPtr = buffer.data.data() + bufferView.byteOffset + attribAccessor.byteOffset;
                    //const size_t count = attribAccessor.count;

                    const size_t byteOffset = bufferView.byteOffset + attribAccessor.byteOffset;
                    const int byteStride = attribAccessor.ByteStride(bufferView);
                    const size_t count = attribAccessor.count;

                    if (attribute.first != "POSITION")
                        continue;

                    // get the position min/max for computing the boundingbox
                    // pMin.x = attribAccessor.minValues[0];
                    // pMin.y = attribAccessor.minValues[1];
                    // pMin.z = attribAccessor.minValues[2];
                    // pMax.x = attribAccessor.maxValues[0];
                    // pMax.y = attribAccessor.maxValues[1];
                    // pMax.z = attribAccessor.maxValues[2];

                    std::vector<glm::vec3> positions;

                    switch (attribAccessor.type)
                    {
                    case TINYGLTF_TYPE_VEC3:
                    {
                        switch (attribAccessor.componentType)
                        {
                        case TINYGLTF_COMPONENT_TYPE_FLOAT:
                            // 3D vector of float
                            // v3fArray positions(
                            //     arrayAdapter<v3f>(dataPtr, count, byte_stride));

                            positions = castBufferToVector<glm::vec3, glm::vec3>(
                                buffer.data.data(), byteOffset, count, byteStride);

                            for (size_t i = 0; i < positions.size(); ++i)
                            {
                                triangleVertices.push_back(positions[i]);
                                //loadedMesh.vertices.push_back(v.x * scale);
                                //loadedMesh.vertices.push_back(v.y * scale);
                                //loadedMesh.vertices.push_back(v.z * scale);
                            }
                        }
                        break;
                    case TINYGLTF_COMPONENT_TYPE_DOUBLE:
                    {
                        std::cout << "Type is DOUBLE\n";
                        switch (attribAccessor.type)
                        {
                        case TINYGLTF_TYPE_VEC3:
                        {
                            // v3dArray positions(
                            //     arrayAdapter<v3d>(dataPtr, count, byte_stride));

                            positions = castBufferToVector<glm::dvec3, glm::vec3>(
                                buffer.data.data(), byteOffset, count, byteStride);

                            for (size_t i = 0; i < positions.size(); ++i)
                            {
                                triangleVertices.push_back(positions[i]);

                                //loadedMesh.vertices.push_back(v.x * scale);
                                //loadedMesh.vertices.push_back(v.y * scale);
                                //loadedMesh.vertices.push_back(v.z * scale);
                            }
                        }
                        break;
                        default:
                            // TODO Handle error
                            break;
                        }
                        break;
                    default:
                        break;
                    }
                    }
                    break;
                    }

                    /*
                    if (attribute.first == "NORMAL")
                    {
                        std::cout << "found normal attribute\n";

                        switch (attribAccessor.type)
                        {
                        case TINYGLTF_TYPE_VEC3:
                        {
                            std::cout << "Normal is VEC3\n";
                            switch (attribAccessor.componentType)
                            {
                            case TINYGLTF_COMPONENT_TYPE_FLOAT:
                            {
                                std::cout << "Normal is FLOAT\n";
                                v3fArray normals(
                                    arrayAdapter<v3f>(dataPtr, count, byte_stride));

                                // IMPORTANT: We need to reorder normals (and texture
                                // coordinates into "facevarying" order) for each face

                                // For each triangle :
                                for (size_t i{0}; i < indices.size() / 3; ++i)
                                {
                                    // get the i'th triange's indexes
                                    auto f0 = indices[3 * i + 0];
                                    auto f1 = indices[3 * i + 1];
                                    auto f2 = indices[3 * i + 2];

                                    // get the 3 normal vectors for that face
                                    v3f n0, n1, n2;
                                    n0 = normals[f0];
                                    n1 = normals[f1];
                                    n2 = normals[f2];

                                    // Put them in the array in the correct order
                                    loadedMesh.facevarying_normals.push_back(n0.x);
                                    loadedMesh.facevarying_normals.push_back(n0.y);
                                    loadedMesh.facevarying_normals.push_back(n0.z);

                                    loadedMesh.facevarying_normals.push_back(n1.x);
                                    loadedMesh.facevarying_normals.push_back(n1.y);
                                    loadedMesh.facevarying_normals.push_back(n1.z);

                                    loadedMesh.facevarying_normals.push_back(n2.x);
                                    loadedMesh.facevarying_normals.push_back(n2.y);
                                    loadedMesh.facevarying_normals.push_back(n2.z);
                                }
                            }
                            break;
                            case TINYGLTF_COMPONENT_TYPE_DOUBLE:
                            {
                                std::cout << "Normal is DOUBLE\n";
                                v3dArray normals(
                                    arrayAdapter<v3d>(dataPtr, count, byte_stride));

                                // IMPORTANT: We need to reorder normals (and texture
                                // coordinates into "facevarying" order) for each face

                                // For each triangle :
                                for (size_t i{0}; i < indices.size() / 3; ++i)
                                {
                                    // get the i'th triange's indexes
                                    auto f0 = indices[3 * i + 0];
                                    auto f1 = indices[3 * i + 1];
                                    auto f2 = indices[3 * i + 2];

                                    // get the 3 normal vectors for that face
                                    v3d n0, n1, n2;
                                    n0 = normals[f0];
                                    n1 = normals[f1];
                                    n2 = normals[f2];

                                    // Put them in the array in the correct order
                                    loadedMesh.facevarying_normals.push_back(n0.x);
                                    loadedMesh.facevarying_normals.push_back(n0.y);
                                    loadedMesh.facevarying_normals.push_back(n0.z);

                                    loadedMesh.facevarying_normals.push_back(n1.x);
                                    loadedMesh.facevarying_normals.push_back(n1.y);
                                    loadedMesh.facevarying_normals.push_back(n1.z);

                                    loadedMesh.facevarying_normals.push_back(n2.x);
                                    loadedMesh.facevarying_normals.push_back(n2.y);
                                    loadedMesh.facevarying_normals.push_back(n2.z);
                                }
                            }
                            break;
                            default:
                                std::cerr << "Unhandeled componant type for normal\n";
                            }
                        }
                        break;
                        default:
                            std::cerr << "Unhandeled vector type for normal\n";
                        }

                        // Face varying comment on the normals is also true for the UVs
                        if (attribute.first == "TEXCOORD_0")
                        {
                            std::cout << "Found texture coordinates\n";

                            switch (attribAccessor.type)
                            {
                            case TINYGLTF_TYPE_VEC2:
                            {
                                std::cout << "TEXTCOORD is VEC2\n";
                                switch (attribAccessor.componentType)
                                {
                                case TINYGLTF_COMPONENT_TYPE_FLOAT:
                                {
                                    std::cout << "TEXTCOORD is FLOAT\n";
                                    v2fArray uvs(
                                        arrayAdapter<v2f>(dataPtr, count, byte_stride));

                                    for (size_t i{0}; i < indices.size() / 3; ++i)
                                    {
                                        // get the i'th triange's indexes
                                        auto f0 = indices[3 * i + 0];
                                        auto f1 = indices[3 * i + 1];
                                        auto f2 = indices[3 * i + 2];

                                        // get the texture coordinates for each triangle's
                                        // vertices
                                        v2f uv0, uv1, uv2;
                                        uv0 = uvs[f0];
                                        uv1 = uvs[f1];
                                        uv2 = uvs[f2];

                                        // push them in order into the mesh data
                                        loadedMesh.facevarying_uvs.push_back(uv0.x);
                                        loadedMesh.facevarying_uvs.push_back(uv0.y);

                                        loadedMesh.facevarying_uvs.push_back(uv1.x);
                                        loadedMesh.facevarying_uvs.push_back(uv1.y);

                                        loadedMesh.facevarying_uvs.push_back(uv2.x);
                                        loadedMesh.facevarying_uvs.push_back(uv2.y);
                                    }
                                }
                                break;
                                case TINYGLTF_COMPONENT_TYPE_DOUBLE:
                                {
                                    std::cout << "TEXTCOORD is DOUBLE\n";
                                    v2dArray uvs(
                                        arrayAdapter<v2d>(dataPtr, count, byte_stride));

                                    for (size_t i{0}; i < indices.size() / 3; ++i)
                                    {
                                        // get the i'th triange's indexes
                                        auto f0 = indices[3 * i + 0];
                                        auto f1 = indices[3 * i + 1];
                                        auto f2 = indices[3 * i + 2];

                                        v2d uv0, uv1, uv2;
                                        uv0 = uvs[f0];
                                        uv1 = uvs[f1];
                                        uv2 = uvs[f2];

                                        loadedMesh.facevarying_uvs.push_back(uv0.x);
                                        loadedMesh.facevarying_uvs.push_back(uv0.y);

                                        loadedMesh.facevarying_uvs.push_back(uv1.x);
                                        loadedMesh.facevarying_uvs.push_back(uv1.y);

                                        loadedMesh.facevarying_uvs.push_back(uv2.x);
                                        loadedMesh.facevarying_uvs.push_back(uv2.y);
                                    }
                                }
                                break;
                                default:
                                    std::cerr << "unrecognized vector type for UV";
                                }
                            }
                            break;
                            default:
                                std::cerr << "unreconized componant type for UV";
                            }
                        }
                    }*/
                }
                break;

            default:
                std::cerr << "primitive mode not implemented";
                break;

                // These aren't triangles:
            case TINYGLTF_MODE_POINTS:
            case TINYGLTF_MODE_LINE:
            case TINYGLTF_MODE_LINE_LOOP:
                std::cerr << "primitive is not triangle based, ignoring";
            }
            }

            // bbox :
            /*v3f bCenter;
            bCenter.x = 0.5f * (pMax.x - pMin.x) + pMin.x;
            bCenter.y = 0.5f * (pMax.y - pMin.y) + pMin.y;
            bCenter.z = 0.5f * (pMax.z - pMin.z) + pMin.z;

            for (size_t v = 0; v < loadedMesh.vertices.size() / 3; v++)
            {
                loadedMesh.vertices[3 * v + 0] -= bCenter.x;
                loadedMesh.vertices[3 * v + 1] -= bCenter.y;
                loadedMesh.vertices[3 * v + 2] -= bCenter.z;
            }

            loadedMesh.pivot_xform[0][0] = 1.0f;
            loadedMesh.pivot_xform[0][1] = 0.0f;
            loadedMesh.pivot_xform[0][2] = 0.0f;
            loadedMesh.pivot_xform[0][3] = 0.0f;

            loadedMesh.pivot_xform[1][0] = 0.0f;
            loadedMesh.pivot_xform[1][1] = 1.0f;
            loadedMesh.pivot_xform[1][2] = 0.0f;
            loadedMesh.pivot_xform[1][3] = 0.0f;

            loadedMesh.pivot_xform[2][0] = 0.0f;
            loadedMesh.pivot_xform[2][1] = 0.0f;
            loadedMesh.pivot_xform[2][2] = 1.0f;
            loadedMesh.pivot_xform[2][3] = 0.0f;

            loadedMesh.pivot_xform[3][0] = bCenter.x;
            loadedMesh.pivot_xform[3][1] = bCenter.y;
            loadedMesh.pivot_xform[3][2] = bCenter.z;
            loadedMesh.pivot_xform[3][3] = 1.0f;

            // TODO handle materials
            for (size_t i{0}; i < loadedMesh.faces.size(); ++i)
                loadedMesh.material_ids.push_back(materials->at(0).id);*/

                //meshes->push_back(loadedMesh);
                //ret = true;
        }

        for (int i = 0; i < triangleIndices.size(); ++i) {
            triangles.push_back(Triangle{
                triangleVertices[triangleIndices[i].v1],
                triangleVertices[triangleIndices[i].v2],
                triangleVertices[triangleIndices[i].v3]
            });
        }

        // build geoms object
        Geom newGeom;

        newGeom.type = MESH;
        newGeom.translation = translation;
        newGeom.rotation = rotation;
        newGeom.scale = scale;
        newGeom.transform = utilityCore::buildTransformationMatrix(newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
        newGeom.materialid = materialID;
        newGeom.triangleStartIdx = startIndex;
        newGeom.triangleCount = triangles.size() - startIndex;
        startIndex = triangles.size();

        geoms.push_back(newGeom);
    }

    //Triangle* triangleBuffer = new Triangle[triangleIndex + 1];

    //// build triangle buffer
    //int globalIndex = 0;
    //for (int i = 0; i < meshTriangleVector.size(); ++i)
    //{
    //    int numTriangles = meshTriangleVector[i].size();
    //    for (int j = 0; j < numTriangles; ++j)
    //    {
    //        triangleBuffer[globalIndex + j] = meshTriangleVector[i][j];
    //    }
    //    globalIndex += numTriangles;
    //}

    //delete[] triangleBuffer;

    /*/ Iterate through all texture declaration in glTF file
    for (const auto &gltfTexture : model.textures)
    {
        std::cout << "Found texture!";
        Texture loadedTexture;
        const auto &image = model.images[gltfTexture.source];
        loadedTexture.components = image.component;
        loadedTexture.width = image.width;
        loadedTexture.height = image.height;

        const auto size =
            image.component * image.width * image.height * sizeof(unsigned char);
        loadedTexture.image = new unsigned char[size];
        memcpy(loadedTexture.image, image.image.data(), size);
        textures->push_back(loadedTexture);
    }*/
    return 0;
}
// namespace example