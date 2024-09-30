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
			newMaterial.hasReflective = 1.0f; // TODO: handle this based on roughness
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
        glm::vec3 scaling = glm::vec3(scale[0], scale[1], scale[2]) / 2.0f;
        if (type == "cube")
        {
            createCube(MatNameToID[p["MATERIAL"]], translation, rotation, scaling);

        }
        else
        {
			createSphere(MatNameToID[p["MATERIAL"]], translation, rotation, scaling);
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
	Geom cube;
	cube.type = MESH;
	cube.materialid = materialid;
    Scene::updateTransform(cube, translation, rotation, scale);

	// update triangles information
	cube.triangleStartIdx = triangles.size();
	cube.triangleEndIdx = triangles.size() + 12;
	// 12 triangles
	glm::vec3 vertices[8] = {
		glm::vec3(-1, -1, -1),
		glm::vec3(-1, -1, 1),
		glm::vec3(-1, 1, -1),
		glm::vec3(-1, 1, 1),
		glm::vec3(1, -1, -1),
		glm::vec3(1, -1, 1),
		glm::vec3(1, 1, -1),
		glm::vec3(1, 1, 1)
	};
    for (int i = 0; i < 8; ++i)
    {
        printf("v[%d] = %s\n", i, glm::to_string(vertices[i]).c_str());
    }
	glm::vec3 normals[6] = {
		glm::vec3(0, 0, -1),
		glm::vec3(0, 0, 1),
		glm::vec3(0, -1, 0),
		glm::vec3(0, 1, 0),
		glm::vec3(-1, 0, 0),
		glm::vec3(1, 0, 0)
	};
	glm::vec2 uvs[4] = {
		glm::vec2(0, 0),
		glm::vec2(0, 1),
		glm::vec2(1, 0),
		glm::vec2(1, 1)
	};
	int indices[36] = {
		0, 1, 2, 2, 1, 3,
		4, 6, 5, 5, 6, 7,
		0, 4, 1, 1, 4, 5,
		2, 3, 6, 6, 3, 7,
		0, 2, 4, 4, 2, 6,
		1, 5, 3, 3, 5, 7
	};
	for (int i = 0; i < 36; i += 3)
	{
		Triangle tri;
		for (int j = 0; j < 3; j++)
		{
			tri.vertices[j] = vertices[indices[i + j]];
			tri.normals[j] = normals[i / 6];
			tri.uvs[j] = uvs[indices[i + j] % 4];
		}
        triangles.push_back(std::move(tri));
	}

    //print translate, rotate, scale
    printf("translate: %s\n", glm::to_string(translation).c_str());
    printf("rotate: %s\n", glm::to_string(rotation).c_str());
    printf("scale: %s\n", glm::to_string(scale).c_str());
    // print transform matrix

	printf("transform matrix: %s\n\n\n\n\n\n", glm::to_string(cube.transform).c_str());
	updateTriangleTransform(cube, triangles);
    geoms.push_back(std::move(cube));

}

void Scene::createSphere(uint32_t materialid, glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale, int latitudeSegments, int longitudeSegments)
{
    printf("create sphere\n");
    Geom sphere;
    sphere.type = MESH;
    sphere.materialid = materialid;
    Scene::updateTransform(sphere, translation, rotation, scale);

    // 更新三角形索引起始位置
    sphere.triangleStartIdx = triangles.size();

    // 定义球体的细分参数
    const int stacks = 20; // 纬度方向细分
    const int slices = 40; // 经度方向细分

    // 存储顶点、法线和纹理坐标
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;

    // 生成顶点数据
    for (int i = 0; i <= stacks; ++i)
    {
        float V = i / (float)stacks;
        float phi = glm::pi<float>() * (-0.5f + V); // 纬度角从 -π/2 到 π/2

        float cosPhi = cos(phi);
        float sinPhi = sin(phi);

        for (int j = 0; j <= slices; ++j)
        {
            float U = j / (float)slices;
            float theta = glm::two_pi<float>() * U; // 经度角从 0 到 2π

            float cosTheta = cos(theta);
            float sinTheta = sin(theta);

            glm::vec3 vertex = glm::vec3(cosPhi * cosTheta, sinPhi, cosPhi * sinTheta);
            vertices.push_back(vertex);

            normals.push_back(glm::normalize(vertex));
            uvs.push_back(glm::vec2(U, V));
        }
    }

    // 生成三角形索引
    for (int i = 0; i < stacks; ++i)
    {
        for (int j = 0; j < slices; ++j)
        {
            int first = i * (slices + 1) + j;
            int second = first + slices + 1;

            // 注意顶点顺序，采用逆时针方向

            // 第一个三角形
            Triangle tri1;
            tri1.vertices[0] = vertices[first];
            tri1.vertices[1] = vertices[second];
            tri1.vertices[2] = vertices[first + 1];

            tri1.normals[0] = normals[first];
            tri1.normals[1] = normals[second];
            tri1.normals[2] = normals[first + 1];

            tri1.uvs[0] = uvs[first];
            tri1.uvs[1] = uvs[second];
            tri1.uvs[2] = uvs[first + 1];

            triangles.push_back(std::move(tri1));

            // 第二个三角形
            Triangle tri2;
            tri2.vertices[0] = vertices[first + 1];
            tri2.vertices[1] = vertices[second];
            tri2.vertices[2] = vertices[second + 1];

            tri2.normals[0] = normals[first + 1];
            tri2.normals[1] = normals[second];
            tri2.normals[2] = normals[second + 1];

            tri2.uvs[0] = uvs[first + 1];
            tri2.uvs[1] = uvs[second];
            tri2.uvs[2] = uvs[second + 1];

            triangles.push_back(std::move(tri2));
        }
    }

    // 更新三角形索引结束位置
    sphere.triangleEndIdx = triangles.size();

    // 输出变换信息
    printf("translate: %s\n", glm::to_string(translation).c_str());
    printf("rotate: %s\n", glm::to_string(rotation).c_str());
    printf("scale: %s\n", glm::to_string(scale).c_str());
    printf("transform matrix: %s\n\n\n\n\n\n", glm::to_string(sphere.transform).c_str());

	updateTriangleTransform(sphere, triangles);
    geoms.push_back(std::move(sphere));
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
    // print attrib vertices
	for (int i = 0; i < attrib.vertices.size(); i += 3)
	{
		printf("v[%d] = %s\n", i / 3, glm::to_string(glm::vec3(attrib.vertices[i], attrib.vertices[i + 1], attrib.vertices[i + 2])).c_str());
	}
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
				tri.normals[v] = glm::vec3(
					attrib.normals[3 * idx.normal_index + 0],
					attrib.normals[3 * idx.normal_index + 1],
					attrib.normals[3 * idx.normal_index + 2]
				);
				tri.uvs[v] = glm::vec2(
					attrib.texcoords[2 * idx.texcoord_index + 0],
					attrib.texcoords[2 * idx.texcoord_index + 1]
				);
			}
            triangles.push_back(std::move(tri));
			newMesh.triangleEndIdx++;
		}
        //print translate, rotate, scale
        printf("translate: %s\n", glm::to_string(translation).c_str());
        printf("rotate: %s\n", glm::to_string(rotation).c_str());
        printf("scale: %s\n", glm::to_string(scale).c_str());
		// print transform matrix
		printf("transform matrix: %s\n", glm::to_string(newMesh.transform).c_str());
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
    bvh = new BVHAccel(this->triangles, this->triangles.size(), 4);
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