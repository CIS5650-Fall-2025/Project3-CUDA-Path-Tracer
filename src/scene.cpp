#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
using json = nlohmann::json;
using Model = tinygltf::Model;
using TinyGLTF = tinygltf::TinyGLTF;

Model model;
TinyGLTF loader;

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
	else if (ext == ".gltf" || ext == ".glb")
	{
		loadFromGltf(filename);
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
			newMaterial.hasReflective = true;
        }
		else if (p["TYPE"] == "Transmissive") {
			const auto& col = p["RGB"];
			newMaterial.color = glm::vec3(col[0], col[1], col[2]);
			newMaterial.indexOfRefraction = p["IOR"];
			newMaterial.hasRefractive = true;
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
        else
        {
            newGeom.type = SPHERE;
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

	camera.apertureRadius = cameraData["APERTURE_RADIUS"];
	camera.focalLength = cameraData["FOCUS_DISTANCE"];

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

void Scene::loadFromGltf(const std::string& gltfName)
{
	std::string err;
	std::string warn;
	bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, gltfName);
	if (!warn.empty())
	{
		std::cerr << "Warn: " << warn << std::endl;
	}

	if (!err.empty())
	{
		std::cerr << "Err: " << err << std::endl;
	}

	if (!ret)
	{
		std::cerr << "Failed to parse glTF file" << std::endl;
		return;
	}

    Camera& sceneCamera = state.camera;
	const auto& scene = model.scenes[model.defaultScene];

	for (const auto& material : model.materials)
	{
		Material newMaterial;
		const auto& pbr = material.pbrMetallicRoughness;
		newMaterial.color = glm::vec3(pbr.baseColorFactor[0], pbr.baseColorFactor[1], pbr.baseColorFactor[2]);
		
        const auto findReflectiveAttribute = material.extensions.find("KHR_materials_pbrSpecularGlossiness");
		newMaterial.hasReflective = findReflectiveAttribute != material.extensions.end();

		const auto findRefractiveAttribute = material.extensions.find("KHR_materials_transmission");
		newMaterial.hasRefractive = findRefractiveAttribute != material.extensions.end();
        const auto findIorAttribute = material.extensions.find("KHR_materials_ior");
		newMaterial.indexOfRefraction = findIorAttribute != material.extensions.end() ? findIorAttribute->second.Get("ior").GetNumberAsDouble() : 1.0f;
		
		const auto findEmissiveAttribute = material.extensions.find("KHR_materials_emissive_strength");
		newMaterial.emittance = findEmissiveAttribute != material.extensions.end() ? findEmissiveAttribute->second.Get("emissiveStrength").GetNumberAsDouble() : 0.0f;

		materials.push_back(newMaterial);
	}

	for (const auto& node : scene.nodes)
	{
		const auto& n = model.nodes[node];
        Geom newGeom;
		newGeom.translation = (n.translation.size() == 0) ? glm::vec3(0.0f) : glm::vec3(n.translation[0], n.translation[1], n.translation[2]);
        if (n.rotation.size() == 4) {
			// GLM expects the quaternion in the order w, x, y, z, whereas GLTF provides it in the order x, y, z, w
            glm::quat quaternion(n.rotation[3], n.rotation[0], n.rotation[1], n.rotation[2]);
            newGeom.rotation = glm::degrees(glm::eulerAngles(quaternion));
        }
        else {
            newGeom.rotation = glm::vec3(0.0f); // Default rotation
        }		
        newGeom.scale = (n.scale.size() == 0) ? glm::vec3(1.0f) : glm::vec3(n.scale[0], n.scale[1], n.scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);

		if (n.mesh >= 0)
		{
            newGeom.type = MESH;
			newGeom.meshId = meshes.size();
            Material newMaterial;
            Mesh newMesh;
            newMesh.vertStartIndex = vertices.size();
			newMesh.trianglesStartIndex = triangles.size();

            const auto& mesh = model.meshes[n.mesh];
			for (const auto& primitive : mesh.primitives)
			{
				parsePrimitive(model, primitive, newMesh);
				// For now, we assume that each primitive has the same material
				newGeom.materialid = primitive.material;
			}
            
			geoms.push_back(newGeom);
			meshes.push_back(newMesh);
		}

		if (n.camera >= 0) {
			const auto& camera = model.cameras[n.camera];
			const auto& perspective = camera.perspective;
			sceneCamera.fov = glm::vec2(glm::degrees(perspective.yfov) * perspective.aspectRatio, glm::degrees(perspective.yfov));
			sceneCamera.position = glm::vec3(n.translation[0], n.translation[1], n.translation[2]);
			glm::quat quaternion = (n.rotation.size() == 4) ? glm::quat(n.rotation[3], n.rotation[0], n.rotation[1], n.rotation[2]) : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
			glm::vec3 forward = quaternion * glm::vec3(0.0f, 0.0f, -1.0f); // note that GLM overloads the * operator for quaternions, so this is effectively q_inv * v * q
            sceneCamera.lookAt = sceneCamera.position + forward;
            sceneCamera.view = glm::normalize(forward);
            sceneCamera.right = glm::normalize(quaternion * glm::vec3(1.0f, 0.0f, 0.0f));
            sceneCamera.up = glm::normalize(quaternion * glm::vec3(0.0f, 1.0f, 0.0f));
        }
	}

    // For now, just hard code other aspects of the scene
    sceneCamera.resolution.x = 800;
    sceneCamera.resolution.y = 800;
    state.iterations = 1000;
    state.traceDepth = 8;
    state.imageName = "cornellgltf";

    float yscaled = tan(sceneCamera.fov.y * (PI / 180));
    float xscaled = (yscaled * sceneCamera.resolution.x) / sceneCamera.resolution.y;
    sceneCamera.pixelLength = glm::vec2(2 * xscaled / (float)sceneCamera.resolution.x,
        2 * yscaled / (float)sceneCamera.resolution.y);

    sceneCamera.apertureRadius = 0.0;
    sceneCamera.focalLength = 10.5;

    //set up render camera stuff
    int arraylen = sceneCamera.resolution.x * sceneCamera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

}

void Scene::parsePrimitive(const Model& model, const tinygltf::Primitive& primitive, Mesh& mesh)
{
    // Access the position attribute
    const auto& posAccessor = model.accessors[primitive.attributes.find("POSITION")->second];
    const auto& posBufferView = model.bufferViews[posAccessor.bufferView];
    const auto& posBuffer = model.buffers[posBufferView.buffer];
    const float* posData = reinterpret_cast<const float*>(&posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset]);

    // Access the normal attribute
    const auto& normAccessor = model.accessors[primitive.attributes.find("NORMAL")->second];
    const auto& normBufferView = model.bufferViews[normAccessor.bufferView];
    const auto& normBuffer = model.buffers[normBufferView.buffer];
    const float* normData = reinterpret_cast<const float*>(&normBuffer.data[normBufferView.byteOffset + normAccessor.byteOffset]);

    // Populate vertices and normals
    for (int i = 0; i < posAccessor.count; ++i)
    {
        vertices.push_back(glm::vec3(posData[i * 3], posData[i * 3 + 1], posData[i * 3 + 2]));
        normals.push_back(glm::vec3(normData[i * 3], normData[i * 3 + 1], normData[i * 3 + 2]));

		mesh.boundingBoxMax = glm::max(mesh.boundingBoxMax, vertices.back());
		mesh.boundingBoxMin = glm::min(mesh.boundingBoxMin, vertices.back());
    }

    // Access the indices
    const auto& indexAccessor = model.accessors[primitive.indices];
    const auto& indexBufferView = model.bufferViews[indexAccessor.bufferView];
    const auto& indexBuffer = model.buffers[indexBufferView.buffer];
    const unsigned short* indexData = reinterpret_cast<const unsigned short*>(&indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset]);

    // Populate triangles
    for (size_t i = 0; i < indexAccessor.count; i += 3)
    {
        Triangle triangle;
        for (int j = 0; j < 3; ++j)
        {
            triangle.vertexIndices[j] = indexData[i + j];
            triangle.normalIndices[j] = indexData[i + j];
        }
        triangles.push_back(triangle);
    }

	mesh.numTriangles = indexAccessor.count / 3;
}
