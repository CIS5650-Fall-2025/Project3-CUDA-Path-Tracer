#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
using json = nlohmann::json;

Scene::Scene(string filename) : envMap(nullptr), bvh(nullptr)
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

Scene::~Scene()
{
	delete envMap;
	envMap = nullptr;
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

		// look at each property of material, check if it exist in the json, if not, use default value
		if (p.contains("RGB"))
		{
			const auto& col = p["RGB"];
			newMaterial.color = glm::vec3(col[0], col[1], col[2]);
		}
		if (p.contains("SPECULAR")) // not correct
		{
			const auto& col = p["SPECULAR"];
			newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
			newMaterial.specular.exponent = p["EXPONENT"];
		}
        if (p.contains("REFLECTIVE"))
        {
			newMaterial.reflective = p["REFLECTIVE"];
        }
		if (p.contains("REFRACTIVE"))
		{
			newMaterial.refractive = p["REFRACTIVE"];
		}
		if (p.contains("IOR"))
		{
			newMaterial.ior = p["IOR"];
		}
		if (p.contains("EMITTANCE"))
		{
			newMaterial.emittance = p["EMITTANCE"];
		}
		if (p.contains("ROUGHNESS"))
		{
			newMaterial.roughness = p["ROUGHNESS"];
		}
        if (p.contains("METALLIC"))
        {
            newMaterial.metallic = p["METALLIC"];
        }

        MatNameToID[name] = materials.size();
        addMaterial(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        glm::vec3 translation = glm::vec3(trans[0], trans[1], trans[2]);
        glm::vec3 rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        glm::vec3 scaling = glm::vec3(scale[0], scale[1], scale[2]);
        if (type == "cube")
        {
            createCube(MatNameToID[p["MATERIAL"]], translation, rotation, scaling);

        }
		else if (type == "sphere")
        {
			createSphere(MatNameToID[p["MATERIAL"]], translation, rotation, scaling);
		}
		else if (type == "mesh")
		{
			const auto& filename = p["FILENAME"];
			loadObj(filename, MatNameToID[p["MATERIAL"]], translation, rotation, scaling);
		}
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

void Scene::createCube(uint32_t materialid, glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale)
{
    printf("create cube\n");

	loadObj("D:/Fall2024/CIS5650/Project3-CUDA-Path-Tracer/scenes/objs/cube.obj", materialid, translation, rotation, scale);
}

void Scene::createSphere(uint32_t materialid, glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale, int latitudeSegments, int longitudeSegments)
{
    printf("create sphere\n");
	loadObj("D:/Fall2024/CIS5650/Project3-CUDA-Path-Tracer/scenes/objs/sphere.obj", materialid, translation, rotation, scale);
}


void Scene::loadObj(const std::string& filename, uint32_t materialid, glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale)
{
	
	printf("load obj\n");
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> meshMaterials;
    //print material info

	std::string warn, err;
	if (!tinyobj::LoadObj(&attrib, &shapes, &meshMaterials, &warn, &err, filename.c_str()))
	{
		throw std::runtime_error(warn + err);
	}
    printf("material size: %d\n", meshMaterials.size());

 //   // print attrib vertices
	//for (int i = 0; i < attrib.vertices.size(); i += 3)
	//{
	//	printf("v[%d] = %s\n", i / 3, glm::to_string(glm::vec3(attrib.vertices[i], attrib.vertices[i + 1], attrib.vertices[i + 2])).c_str());
	//}
	for (const auto& shape : shapes)
	{
		Geom newMesh;
		newMesh.type = MESH;
		newMesh.materialid = materialid;
        Scene::updateTransform(newMesh, translation, rotation, scale);
		

        if (shape.mesh.num_face_vertices[0] != 3)
        {
            throw std::runtime_error("Only triangles are supported");
        }

		newMesh.triangleStartIdx = triangles.size();
        newMesh.triangleEndIdx = triangles.size();
        // assume only triangles
		for (size_t f = 0; f < shape.mesh.indices.size(); f += 3)
		{
			Triangle tri;
			for (size_t v = 0; v < 3; v++)
			{
				const auto& idx = shape.mesh.indices[f + v];
				tri.vertices[v] = glm::vec3(
					attrib.vertices[3 * idx.vertex_index + 0],
					attrib.vertices[3 * idx.vertex_index + 1],
					attrib.vertices[3 * idx.vertex_index + 2]
				);
				if (attrib.normals.size() > 0)
				{
					tri.normals[v] = glm::vec3(
						attrib.normals[3 * idx.normal_index + 0],
						attrib.normals[3 * idx.normal_index + 1],
						attrib.normals[3 * idx.normal_index + 2]
					);
				}
				if (attrib.texcoords.size() > 0)
				{
					tri.uvs[v] = glm::vec2(
						attrib.texcoords[2 * idx.texcoord_index + 0],
						attrib.texcoords[2 * idx.texcoord_index + 1]
					);
				}
				tri.hasNormals = attrib.normals.size() > 0;

			}
            triangles.push_back(std::move(tri));
			newMesh.triangleEndIdx++;
		}
		printf("Loaded %s with %d triangles\n\n\n\n\n", filename.c_str(), newMesh.triangleEndIdx - newMesh.triangleStartIdx);

		updateTriangleTransform(newMesh, triangles);
		geoms.push_back(newMesh);
	}
}

void Scene::addMaterial(Material& m)
{
	m.materialId = materials.size();
	materials.push_back(m);
}

void Scene::loadEnvMap(const char* filename)
{
    if (envMap)
        delete envMap;
    envMap = nullptr;
	envMap = new Texture(filename);
	printf("Loaded environment map %s\n", filename);
    //printf("cuda texture object created: %d\n", envMap->texObj);
}

void Scene::updateTransform(Geom& geom, glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale)
{
    geom.translation = translation;
    geom.rotation = rotation;
    geom.scale = scale;
    geom.transform = utilityCore::buildTransformationMatrix(
        geom.translation, geom.rotation, geom.scale);
    geom.inverseTransform = glm::inverse(geom.transform);
    geom.invTranspose = glm::inverseTranspose(geom.transform);
}

void Scene::updateTriangleTransform(const Geom& geom, std::vector<Triangle>& triangles)
{
    for (int i = geom.triangleStartIdx; i < geom.triangleEndIdx; ++i)
    {
        auto& tri = triangles[i];
        for (int j = 0; j < 3; ++j)
        {
            tri.vertices[j] = glm::vec3(geom.transform * glm::vec4(tri.vertices[j], 1.0f));
            tri.normals[j] = glm::normalize(glm::vec3(geom.invTranspose * glm::vec4(tri.normals[j], 0.0f)));
        }
		tri.materialid = geom.materialid;
    }
}


void Scene::createBVH()
{
    if (bvh != nullptr)
    {
        delete bvh;
    }
    bvh = new BVHAccel(this->triangles, this->triangles.size(), 6);
	bvh->build(this->triangles, this->triangles.size());
    printf("BVH created\n");
}

BVHAccel::LinearBVHNode* Scene::getLBVHRoot()
{
	if (bvh == nullptr)
	{
		printf("BVH not created\n");
		return nullptr;
	}
	return bvh->nodes;
}