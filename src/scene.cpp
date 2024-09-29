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

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

static void PrintInfo(const tinyobj::attrib_t& attrib,
    const std::vector<tinyobj::shape_t>& shapes,
    const std::vector<tinyobj::material_t>& materials) {
    std::cout << "# of vertices  : " << (attrib.vertices.size() / 3) << std::endl;
    std::cout << "# of normals   : " << (attrib.normals.size() / 3) << std::endl;
    std::cout << "# of texcoords : " << (attrib.texcoords.size() / 2)
        << std::endl;

    std::cout << "# of shapes    : " << shapes.size() << std::endl;
    std::cout << "# of materials : " << materials.size() << std::endl;

    for (size_t v = 0; v < attrib.vertices.size() / 3; v++) {
        printf("  v[%ld] = (%f, %f, %f)\n", static_cast<long>(v),
            static_cast<const double>(attrib.vertices[3 * v + 0]),
            static_cast<const double>(attrib.vertices[3 * v + 1]),
            static_cast<const double>(attrib.vertices[3 * v + 2]));
    }

    for (size_t v = 0; v < attrib.normals.size() / 3; v++) {
        printf("  n[%ld] = (%f, %f, %f)\n", static_cast<long>(v),
            static_cast<const double>(attrib.normals[3 * v + 0]),
            static_cast<const double>(attrib.normals[3 * v + 1]),
            static_cast<const double>(attrib.normals[3 * v + 2]));
    }

    for (size_t v = 0; v < attrib.texcoords.size() / 2; v++) {
        printf("  uv[%ld] = (%f, %f)\n", static_cast<long>(v),
            static_cast<const double>(attrib.texcoords[2 * v + 0]),
            static_cast<const double>(attrib.texcoords[2 * v + 1]));
    }

    // For each shape
    for (size_t i = 0; i < shapes.size(); i++) {
        printf("shape[%ld].name = %s\n", static_cast<long>(i),
            shapes[i].name.c_str());
        printf("Size of shape[%ld].mesh.indices: %lu\n", static_cast<long>(i),
            static_cast<unsigned long>(shapes[i].mesh.indices.size()));
        printf("Size of shape[%ld].lines.indices: %lu\n", static_cast<long>(i),
            static_cast<unsigned long>(shapes[i].lines.indices.size()));
        printf("Size of shape[%ld].points.indices: %lu\n", static_cast<long>(i),
            static_cast<unsigned long>(shapes[i].points.indices.size()));

        size_t index_offset = 0;

        assert(shapes[i].mesh.num_face_vertices.size() ==
            shapes[i].mesh.material_ids.size());

        assert(shapes[i].mesh.num_face_vertices.size() ==
            shapes[i].mesh.smoothing_group_ids.size());

        printf("shape[%ld].num_faces: %lu\n", static_cast<long>(i),
            static_cast<unsigned long>(shapes[i].mesh.num_face_vertices.size()));

        // For each face
        for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++) {
            size_t fnum = shapes[i].mesh.num_face_vertices[f];

            printf("  face[%ld].fnum = %ld\n", static_cast<long>(f),
                static_cast<unsigned long>(fnum));

            // For each vertex in the face
            for (size_t v = 0; v < fnum; v++) {
                tinyobj::index_t idx = shapes[i].mesh.indices[index_offset + v];
                printf("    face[%ld].v[%ld].idx = %d/%d/%d\n", static_cast<long>(f),
                    static_cast<long>(v), idx.vertex_index, idx.normal_index,
                    idx.texcoord_index);
            }

            printf("  face[%ld].material_id = %d\n", static_cast<long>(f),
                shapes[i].mesh.material_ids[f]);
            printf("  face[%ld].smoothing_group_id = %d\n", static_cast<long>(f),
                shapes[i].mesh.smoothing_group_ids[f]);

            index_offset += fnum;
        }

        printf("shape[%ld].num_tags: %lu\n", static_cast<long>(i),
            static_cast<unsigned long>(shapes[i].mesh.tags.size()));
        for (size_t t = 0; t < shapes[i].mesh.tags.size(); t++) {
            printf("  tag[%ld] = %s ", static_cast<long>(t),
                shapes[i].mesh.tags[t].name.c_str());
            printf(" ints: [");
            for (size_t j = 0; j < shapes[i].mesh.tags[t].intValues.size(); ++j) {
                printf("%ld", static_cast<long>(shapes[i].mesh.tags[t].intValues[j]));
                if (j < (shapes[i].mesh.tags[t].intValues.size() - 1)) {
                    printf(", ");
                }
            }
            printf("]");

            printf(" floats: [");
            for (size_t j = 0; j < shapes[i].mesh.tags[t].floatValues.size(); ++j) {
                printf("%f", static_cast<const double>(
                    shapes[i].mesh.tags[t].floatValues[j]));
                if (j < (shapes[i].mesh.tags[t].floatValues.size() - 1)) {
                    printf(", ");
                }
            }
            printf("]");

            printf(" strings: [");
            for (size_t j = 0; j < shapes[i].mesh.tags[t].stringValues.size(); ++j) {
                printf("%s", shapes[i].mesh.tags[t].stringValues[j].c_str());
                if (j < (shapes[i].mesh.tags[t].stringValues.size() - 1)) {
                    printf(", ");
                }
            }
            printf("]");
            printf("\n");
        }
    }

    for (size_t i = 0; i < materials.size(); i++) {
        printf("material[%ld].name = %s\n", static_cast<long>(i),
            materials[i].name.c_str());
        printf("  material.Ka = (%f, %f ,%f)\n",
            static_cast<const double>(materials[i].ambient[0]),
            static_cast<const double>(materials[i].ambient[1]),
            static_cast<const double>(materials[i].ambient[2]));
        printf("  material.Kd = (%f, %f ,%f)\n",
            static_cast<const double>(materials[i].diffuse[0]),
            static_cast<const double>(materials[i].diffuse[1]),
            static_cast<const double>(materials[i].diffuse[2]));
        printf("  material.Ks = (%f, %f ,%f)\n",
            static_cast<const double>(materials[i].specular[0]),
            static_cast<const double>(materials[i].specular[1]),
            static_cast<const double>(materials[i].specular[2]));
        printf("  material.Tr = (%f, %f ,%f)\n",
            static_cast<const double>(materials[i].transmittance[0]),
            static_cast<const double>(materials[i].transmittance[1]),
            static_cast<const double>(materials[i].transmittance[2]));
        printf("  material.Ke = (%f, %f ,%f)\n",
            static_cast<const double>(materials[i].emission[0]),
            static_cast<const double>(materials[i].emission[1]),
            static_cast<const double>(materials[i].emission[2]));
        printf("  material.Ns = %f\n",
            static_cast<const double>(materials[i].shininess));
        printf("  material.Ni = %f\n", static_cast<const double>(materials[i].ior));
        printf("  material.dissolve = %f\n",
            static_cast<const double>(materials[i].dissolve));
        printf("  material.illum = %d\n", materials[i].illum);
        printf("  material.map_Ka = %s\n", materials[i].ambient_texname.c_str());
        printf("  material.map_Kd = %s\n", materials[i].diffuse_texname.c_str());
        printf("  material.map_Ks = %s\n", materials[i].specular_texname.c_str());
        printf("  material.map_Ns = %s\n",
            materials[i].specular_highlight_texname.c_str());
        printf("  material.map_bump = %s\n", materials[i].bump_texname.c_str());
        printf("    bump_multiplier = %f\n", static_cast<const double>(materials[i].bump_texopt.bump_multiplier));
        printf("  material.map_d = %s\n", materials[i].alpha_texname.c_str());
        printf("  material.disp = %s\n", materials[i].displacement_texname.c_str());
        printf("  <<PBR>>\n");
        printf("  material.Pr     = %f\n", static_cast<const double>(materials[i].roughness));
        printf("  material.Pm     = %f\n", static_cast<const double>(materials[i].metallic));
        printf("  material.Ps     = %f\n", static_cast<const double>(materials[i].sheen));
        printf("  material.Pc     = %f\n", static_cast<const double>(materials[i].clearcoat_thickness));
        printf("  material.Pcr    = %f\n", static_cast<const double>(materials[i].clearcoat_roughness));
        printf("  material.aniso  = %f\n", static_cast<const double>(materials[i].anisotropy));
        printf("  material.anisor = %f\n", static_cast<const double>(materials[i].anisotropy_rotation));
        printf("  material.map_Ke = %s\n", materials[i].emissive_texname.c_str());
        printf("  material.map_Pr = %s\n", materials[i].roughness_texname.c_str());
        printf("  material.map_Pm = %s\n", materials[i].metallic_texname.c_str());
        printf("  material.map_Ps = %s\n", materials[i].sheen_texname.c_str());
        printf("  material.norm   = %s\n", materials[i].normal_texname.c_str());
        std::map<std::string, std::string>::const_iterator it(
            materials[i].unknown_parameter.begin());
        std::map<std::string, std::string>::const_iterator itEnd(
            materials[i].unknown_parameter.end());

        for (; it != itEnd; it++) {
            printf("  material.%s = %s\n", it->first.c_str(), it->second.c_str());
        }
        printf("\n");
    }

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
    geoms.push_back(std::move(cube));
}

void Scene::createSphere(Geom& cube, uint32_t materialid, glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale)
{
   
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

void Scene::createBVH()
{
    if (bvh != nullptr)
    {
        delete bvh;
    }
    bvh = new BVHAccel(this->triangles.data(), this->triangles.size(), 4);
	bvh->build(this->triangles.data(), this->triangles.size());
    printf("BVH created\n");
}