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
			const auto& col = p["RGB"];
			newMaterial.color = glm::vec3(col[0], col[1], col[2]);
			newMaterial.hasReflective = 0.0f;
			newMaterial.hasRefractive = 0.0f;
		}
		else if (p["TYPE"] == "Emitting")
		{
			const auto& col = p["RGB"];
			newMaterial.color = glm::vec3(col[0], col[1], col[2]);
			newMaterial.emittance = p["EMITTANCE"];
			newMaterial.hasReflective = 0.0f;
			newMaterial.hasRefractive = 0.0f;
		}
		else if (p["TYPE"] == "Specular")
		{
			const auto& col = p["RGB"];
			float roughness = p["ROUGHNESS"];
			newMaterial.color = glm::vec3(col[0], col[1], col[2]);
			newMaterial.hasReflective = 1.0f - roughness;
			newMaterial.hasRefractive = 0.0f;
		}
		else if (p["TYPE"] == "Transparent")
		{
			const auto& col = p["RGB"];
			float roughness = p["ROUGHNESS"];
			newMaterial.color = glm::vec3(col[0], col[1], col[2]);		
			newMaterial.hasReflective = 1.0f - roughness;
			newMaterial.hasRefractive = p["TRANSPARENCY"];
			newMaterial.indexOfRefraction = p["IOR"];
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

template <typename T, typename U>
std::vector<U> Scene::castBufferToVector(const unsigned char* buffer, size_t count, int stride)
{
	std::vector<U> result;
	result.reserve(count);

	for (size_t i = 0; i < count; ++i)
	{
		T value;
		std::memcpy(&value, &buffer[i * stride], sizeof(T));
		result.push_back(static_cast<U>(value));
	}
	return result;
}

// Heavily adapted from https://github.com/syoyo/tinygltf/blob/release/examples/raytrace/gltf-loader.cc
void Scene::loadGLTF(const std::string& filename, int materialID, const glm::vec3 translation, const glm::vec3 rotation, const glm::vec3 scale) {
	tinygltf::Model model;
	tinygltf::TinyGLTF loader;

	const std::string fileExtension = filename.substr(filename.find_last_of(".") + 1);

	bool successfulLoad = false;
	std::string err;
	std::string warn;
	if (fileExtension == "glb") {
		// Binary input
		successfulLoad = loader.LoadBinaryFromFile(&model, &err, &warn, filename.c_str());
	}
	else if (fileExtension == "gltf") {
		// ASCII input
		successfulLoad = loader.LoadASCIIFromFile(&model, &err, &warn, filename.c_str());
	}

	// Error handling
	if (!warn.empty()) std::cout << "glTF parse warning: " << warn << std::endl;
	if (!err.empty()) std::cerr << "glTF parse error: " << err << std::endl;
	if (!successfulLoad) std::cerr << "Failed to load glTF: " << filename << std::endl;

	int triangleIndex = 0;
	int startIndex = 0;
	// Iterate through all the meshes in the glTF file
	for (const auto& gltfMesh : model.meshes) {
		// For each primitive
		for (const auto& meshPrimitive : gltfMesh.primitives) {
			std::vector<TriangleIdx> triangleIndices;
			std::vector<glm::vec3> triangleVertices;
			std::vector<glm::vec3> triangleNormals;
			std::vector<glm::vec2> triangleUVs;
			glm::vec3 boundingBoxMin;
			glm::vec3 boundingBoxMax;
			float vertexScale;

			const auto& indicesAccessor = model.accessors[meshPrimitive.indices];
			const auto& bufferView = model.bufferViews[indicesAccessor.bufferView];
			const auto& buffer = model.buffers[bufferView.buffer];

			const auto bufferPtr = buffer.data.data() + bufferView.byteOffset + indicesAccessor.byteOffset;
			const int byteStride = indicesAccessor.ByteStride(bufferView);
			const size_t count = indicesAccessor.count;

			std::vector<unsigned int> indicesVector;
			switch (indicesAccessor.componentType) {
			case TINYGLTF_COMPONENT_TYPE_BYTE:
				indicesVector = castBufferToVector<char, unsigned int>(bufferPtr, count, byteStride);
				break;
			case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
				indicesVector = castBufferToVector<unsigned char, unsigned int>(bufferPtr, count, byteStride);
				break;
			case TINYGLTF_COMPONENT_TYPE_SHORT:
				indicesVector = castBufferToVector<short, unsigned unsigned int>(bufferPtr, count, byteStride);
				break;
			case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
				indicesVector = castBufferToVector<unsigned short, unsigned int>(bufferPtr, count, byteStride);
				break;
			case TINYGLTF_COMPONENT_TYPE_INT:
				indicesVector = castBufferToVector<int, unsigned int>(bufferPtr, count, byteStride);
				break;
			case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
				indicesVector = castBufferToVector<unsigned int, unsigned int>(bufferPtr, count, byteStride);
				break;
			default:
				std::cerr << "Unsupported index component type: " << indicesAccessor.componentType << std::endl;
			}

			switch (meshPrimitive.mode) {
			case TINYGLTF_MODE_TRIANGLE_FAN:
				for (size_t i = 2; i < indicesVector.size(); ++i) {
					triangleIndices.push_back(TriangleIdx{ indicesVector[0], indicesVector[i - 1], indicesVector[i] });
				}
				break;
			case TINYGLTF_MODE_TRIANGLE_STRIP:
				for (size_t i = 2; i < indicesVector.size(); ++i) {
					triangleIndices.push_back(TriangleIdx{ indicesVector[i - 2], indicesVector[i - 1], indicesVector[i] });
				}
				break;
			case TINYGLTF_MODE_TRIANGLES:
				for (size_t i = 2; i < indicesVector.size(); i += 3) {
					triangleIndices.push_back(TriangleIdx{ indicesVector[i - 2], indicesVector[i - 1], indicesVector[i] });
				}
				break;
			default:
				std::cerr << "Unsupported primitive mode: " << meshPrimitive.mode << std::endl;
			}

			// Iterating through all the attributes of the primitve
			for (const auto& attribute : meshPrimitive.attributes) {
				const auto& attribAccessor = model.accessors[attribute.second];
				const auto& bufferView = model.bufferViews[attribAccessor.bufferView];
				const auto& buffer = model.buffers[bufferView.buffer];

				const auto bufferPtr = buffer.data.data() + bufferView.byteOffset + attribAccessor.byteOffset;
				const int byteStride = attribAccessor.ByteStride(bufferView);
				const size_t count = attribAccessor.count;

				// Vertex position data
				if (attribute.first == "POSITION") {
					boundingBoxMin = glm::vec3{
						attribAccessor.minValues[0],
						attribAccessor.minValues[1],
						attribAccessor.minValues[2]
					};
					boundingBoxMax = glm::vec3{
						attribAccessor.maxValues[0],
						attribAccessor.maxValues[1],
						attribAccessor.maxValues[2]
					};

					glm::vec3 boundingBoxScale = boundingBoxMax - boundingBoxMin;
					vertexScale = 1.0f / max(max(boundingBoxScale.x, boundingBoxScale.y), boundingBoxScale.z);

					boundingBoxMin *= vertexScale;
					boundingBoxMax *= vertexScale;

					if (attribAccessor.type != TINYGLTF_TYPE_VEC3) {
						std::cerr << "Unsupported position type: " << attribAccessor.type << std::endl;
						continue;
					}

					std::vector<glm::vec3> positions;

					switch (attribAccessor.componentType) {
					case TINYGLTF_COMPONENT_TYPE_FLOAT:
						positions = castBufferToVector<glm::vec3, glm::vec3>(bufferPtr, count, byteStride);

						for (size_t i = 0; i < positions.size(); ++i) {
							triangleVertices.push_back(vertexScale * positions[i]);
						}
						break;
					case TINYGLTF_COMPONENT_TYPE_DOUBLE:
						positions = castBufferToVector<glm::dvec3, glm::vec3>(bufferPtr, count, byteStride);

						for (size_t i = 0; i < positions.size(); ++i) {
							triangleVertices.push_back(vertexScale * positions[i]);
						}
						break;
					default:
						std::cerr << "Unsupported position component type: " << attribAccessor.componentType << std::endl;
					}
				}

				// Vertex normal data
				if (attribute.first == "NORMAL") {
					if (attribAccessor.type != TINYGLTF_TYPE_VEC3) {
						std::cerr << "Unsupported normal type: " << attribAccessor.type << std::endl;
						continue;
					}

					std::vector<glm::vec3> normals;

					switch (attribAccessor.componentType) {
					case TINYGLTF_COMPONENT_TYPE_FLOAT:
						normals = castBufferToVector<glm::vec3, glm::vec3>(bufferPtr, count, byteStride);

						for (size_t i = 0; i < normals.size(); ++i) {
							triangleNormals.push_back(normals[i]);
						}
						break;
					case TINYGLTF_COMPONENT_TYPE_DOUBLE:
						normals = castBufferToVector<glm::dvec3, glm::vec3>(bufferPtr, count, byteStride);

						for (size_t i = 0; i < normals.size(); ++i) {
							triangleNormals.push_back(normals[i]);
						}
						break;
					default:
						std::cerr << "Unsupported normal component type: " << attribAccessor.componentType << std::endl;
					}
				}

				// Vertex UV data
				if (attribute.first == "TEXCOORD_0") {
					if (attribAccessor.type != TINYGLTF_TYPE_VEC2) {
						std::cerr << "Unsupported uv type: " << attribAccessor.type << std::endl;
						continue;
					}

					std::vector<glm::vec2> uvs;

					switch (attribAccessor.componentType) {
					case TINYGLTF_COMPONENT_TYPE_FLOAT:
						uvs = castBufferToVector<glm::vec2, glm::vec2>(bufferPtr, count, byteStride);

						for (size_t i = 0; i < uvs.size(); ++i) {
							triangleUVs.push_back(uvs[i]);
						}
						break;
					case TINYGLTF_COMPONENT_TYPE_DOUBLE:
						uvs = castBufferToVector<glm::dvec2, glm::vec2>(bufferPtr, count, byteStride);

						for (size_t i = 0; i < uvs.size(); ++i) {
							triangleUVs.push_back(uvs[i]);
						}
						break;
					default:
						std::cerr << "Unsupported uv component type: " << attribAccessor.componentType << std::endl;
					}
				}
			}

			if (triangleUVs.size() != triangleVertices.size()) {
				triangleUVs.resize(triangleVertices.size(), glm::vec2(0.0f));
			}

			if (triangleNormals.size() == triangleVertices.size()) {
				for (int i = 0; i < triangleIndices.size(); ++i) {
					triangles.push_back(Triangle{
						triangleVertices[triangleIndices[i].v1],
						triangleVertices[triangleIndices[i].v2],
						triangleVertices[triangleIndices[i].v3],
						triangleNormals[triangleIndices[i].v1],
						triangleNormals[triangleIndices[i].v2],
						triangleNormals[triangleIndices[i].v3],
						triangleUVs[triangleIndices[i].v1],
						triangleUVs[triangleIndices[i].v2],
						triangleUVs[triangleIndices[i].v3]
						});
				}
			}
			else {
				for (int i = 0; i < triangleIndices.size(); ++i) {
					glm::vec3 v1 = triangleVertices[triangleIndices[i].v1];
					glm::vec3 v2 = triangleVertices[triangleIndices[i].v2];
					glm::vec3 v3 = triangleVertices[triangleIndices[i].v3];

					glm::vec3 normal = glm::normalize(glm::cross(v2 - v1, v3 - v1));

					triangles.push_back(Triangle{
						v1, v2, v3,
						normal, normal, normal,
						triangleUVs[triangleIndices[i].v1],
						triangleUVs[triangleIndices[i].v2],
						triangleUVs[triangleIndices[i].v3]
						});
				}
			}

			// build geoms object
			Geom newGeom;
			newGeom.type = MESH;
			newGeom.translation = translation;
			newGeom.rotation = rotation;
			newGeom.scale = scale / vertexScale;
			newGeom.transform = utilityCore::buildTransformationMatrix(newGeom.translation, newGeom.rotation, newGeom.scale);
			newGeom.inverseTransform = glm::inverse(newGeom.transform);
			newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
			newGeom.materialid = materialID;
			newGeom.triangleStartIdx = startIndex;
			newGeom.triangleCount = triangles.size() - startIndex;
			newGeom.boundingBoxMin = boundingBoxMin;
			newGeom.boundingBoxMax = boundingBoxMax;
			startIndex = triangles.size();

			geoms.push_back(newGeom);
		}
	}

	// Iterate through all texture declaration in glTF file
	for (const auto &gltfTexture : model.textures)
	{
		const auto& image = model.images[gltfTexture.source];

		Texture texture;
		texture.width = image.width;
		texture.height = image.height;
		const auto components = image.component;

		texture.data.resize(4 * image.width * image.height);
		switch (image.pixel_type) {
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
			for (int i = 0; i < texture.width * texture.height; ++i) {
				texture.data[4 * i + 0] = image.image[components * i + 0];
				texture.data[4 * i + 1] = image.image[components * i + 1];
				texture.data[4 * i + 2] = image.image[components * i + 2];
				texture.data[4 * i + 3] = 255;
			}
			break;
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
			for (int i = 0; i < texture.width * texture.height; ++i) {
				texture.data[4 * i + 0] = image.image[2 * components * i + 0];
				texture.data[4 * i + 1] = image.image[2 * components * i + 2];
				texture.data[4 * i + 2] = image.image[2 * components * i + 4];
				texture.data[4 * i + 3] = 255;
			}
			break;
		default:
			std::cerr << "Unsupported pixel type: " << image.pixel_type << std::endl;
			continue;
		}


		//if (image.pixel_type != TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE && image.pixel_type != TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
		//	std::cerr << "Unsupported pixel type: " << image.pixel_type << std::endl;
		//	continue;
		//}

		//// Handle 16 bit textures
		//auto convertPixel = [image](unsigned short pixel) {
		//	if (image.pixel_type == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
		//		return (uint8_t)(pixel);
		//	}
		//	else {
		//		return (uint8_t)pixel;
		//	}
		//};

		// Only saving the first 3 components (RGB)
		// Alpha channel is always 255


	    textures.push_back(texture);
	}
}