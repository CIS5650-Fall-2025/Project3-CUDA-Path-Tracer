#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
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

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}





glm::vec2 calculatePixelLength(float xscale, float yscale, glm::ivec2 resolution) {
    float pixelWidth = (2 * xscale) / static_cast<float>(resolution.x);
    float pixelHeight = (2 * yscale) / static_cast<float>(resolution.y);
    return glm::vec2(pixelWidth, pixelHeight);
}


//
// "RES": An array representing the resolution of the output image in pixels.
// "FOVY" : The vertical field of view, in degrees.
// "ITERATIONS" : The number of iterations to refine the image during rendering.
// "DEPTH" : The maximum path tracing depth.
// "FILE" : The filename for the rendered output.
// "EYE" : The position of the camera in world coordinates.
// "LOOKAT" : The point in space the camera is directed at.
// "UP" : The up vector defining the camera's orientation.
void Scene::loadCamera() {
    cout << "Start Loading Camera ..." << endl;
    RenderState& curr_state = this->state;
    Camera& camera = curr_state.camera;
    float fov_y;


    for (int i = 0; i < 5; i++) {
        string curr_line;
        utilityCore::safeGetline(fp_in, curr_line);
        vector<string> tokens = utilityCore::tokenizeString(curr_line);
        if (tokens[0] == "RES") {
            camera.resolution.x = std::stoi(tokens[1]);
            camera.resolution.y = std::stoi(tokens[2]);
        }
        else if (tokens[0] == "FOVY") {
            fov_y = std::stof(tokens[1]);
        }
        else if (tokens[0] == "ITERATIONS") {
            state.iterations = std::stoi(tokens[1]);
        }
        else if (tokens[0] == "DEPTH") {
            state.traceDepth = std::stoi(tokens[1]);
        }
        else if (tokens[0] == "FILE") {
            state.imageName = tokens[1];
        }
        else if (tokens[0] == "EYE") {
            camera.position = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        }
        else if (tokens[0] == "LOOKAT") {
            camera.lookAt = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        }
        else if (tokens[0] == "UP") {
            camera.up = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        }
    }


    //calculate fov_x
    float yscaled = tan(fov_y * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;

    float fov_x = (atan(xscaled) * 180) / PI;

    camera.fov = glm::vec2(fov_x, fov_y);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = calculatePixelLength(xscaled, yscaled, camera.resolution);
    camera.view = glm::normalize(camera.lookAt - camera.position);

    int num_pixels = camera.resolution.x * camera.resolution.y;
    curr_state.image.resize(num_pixels);

    std::fill(curr_state.image.begin(), curr_state.image.end(), glm::vec3());

    cout << "Camera Loaded" << endl;
}



int Scene::constructBVH(int start_index, int end_index) {


    // Create a list of triangle indices
    std::vector<BVH_TriangleAABB_relation> relations(BVH_triangles.size());

    for (int i = 0; i < BVH_triangles.size(); i++) {
        const Triangle& T = BVH_triangles[i];

        const glm::vec3& v1 = T.vertices[0];
        const glm::vec3& v2 = T.vertices[1];
        const glm::vec3& v3 = T.vertices[2];

        relations[i] = BVH_TriangleAABB_relation(i, AABB(v1, v2, v3));

    }

    // In a fully binary tree, the maximum number of nodes is 2n - 1 for n primitives.
    int num_nodes = 2 * (end_index - start_index) - 1;

    std::vector<BVH_Node> nodes(num_nodes);

    BVH_Node& root = nodes[0];
    root.leftChild = 0;
    root.rightChild = 0;
    root.fisrtT_idx = 0;
    root.TCount = end_index - start_index;

    updateBVH(relations, nodes, 0);


    // Start the subdivision process to construct the BVH
    int visited_node = 1; // Start with the root node visited
    int largestLeafSize = 0; // To track the maximum leaf size
    int treeDepth = subdivide(relations, nodes, 0, visited_node, largestLeafSize);

    // Store the constructed nodes in the class's BVH_Nodes vector
    BVH_Nodes.clear(); // Clear any existing BVH nodes
    BVH_Nodes.insert(BVH_Nodes.end(), nodes.begin(), nodes.begin() + visited_node);

    return treeDepth;
}




void Scene::updateBVH(const std::vector<BVH_TriangleAABB_relation>& relation, std::vector<BVH_Node>& nodes, int idx)
{
    // validation check
    if (0 <= idx && idx < nodes.size()) {

        BVH_Node& node = nodes[idx];
        node.aabb = AABB();

        for (int i = node.fisrtT_idx; i < node.fisrtT_idx + node.TCount; i++)
        {
            node.aabb = combine2AABB(node.aabb, relation[i].aabb);
        }
    }
}




// ses Surface Area Heuristic (SAH) to determine the best split point

// relations: A vector containing BVH_TriangleAABB_relation objects, which hold information about triangles and their bounding boxes(AABBs).
// nodes : A vector of BVH_Node objects representing the BVH nodes.
// idx : The index of the current node in the nodes vector.
// visited_node : A reference that tracks the total number of nodes created.
// largestLeafSize : A reference that will store the maximum size of triangle in a node.
int Scene::subdivide(std::vector<BVH_TriangleAABB_relation>& relations, std::vector<BVH_Node>& nodes, int idx, int& visited_node, int& largestLeafSize) {
    
    //validation check
    assert(idx >= 0 && idx < nodes.size());

    BVH_Node& node = nodes[idx];
    node.splitAxis = node.aabb.LongestAxisIndex();
    int axis = node.splitAxis;

    int num_triangles = node.TCount;
    if (num_triangles <= 2) {
        // Create a leaf node if the number of primitives is small
        largestLeafSize = num_triangles;
        return 1;
    }

    // Calculate bounding boxes for SAH
    std::vector<AABB> leftBounds(num_triangles), rightBounds(num_triangles);
    AABB leftBox, rightBox;

    // Sort triangles along the chosen axis
    std::sort(relations.begin() + node.fisrtT_idx, relations.begin() + node.fisrtT_idx + num_triangles,
        [this, axis](const BVH_TriangleAABB_relation& a, const BVH_TriangleAABB_relation& b) {
            auto Ta = BVH_triangles[a.triangle_id];
            auto Tb = BVH_triangles[b.triangle_id];

            const glm::vec3& v1_a = Ta.vertices[0];
            const glm::vec3& v2_a = Ta.vertices[1];
            const glm::vec3& v3_a = Ta.vertices[2];

            const glm::vec3& v1_b = Tb.vertices[0];
            const glm::vec3& v2_b = Tb.vertices[1];
            const glm::vec3& v3_b = Tb.vertices[2];

            glm::vec3 centroid_a = (v1_a + v2_a + v3_a) * (1.0f / 3.0f);
            glm::vec3 centroid_b = (v1_b + v2_b + v3_b) * (1.0f / 3.0f);

            return centroid_a[axis] < centroid_b[axis];
        });
    
    // Compute bounding boxes from left to right
    for (int i = 0; i < num_triangles; i++) {
        leftBox = combine2AABB(leftBox, relations[node.fisrtT_idx + i].aabb);
        leftBounds[i] = leftBox;
    }

    // Compute bounding boxes from right to left
    for (int i = num_triangles - 1; i >= 0; i--) {
        rightBox = combine2AABB(rightBox, relations[node.fisrtT_idx + i].aabb);
        rightBounds[i] = rightBox;
    }

    // Find the best split position using SAH
    float bestCost = std::numeric_limits<float>::infinity();
    int bestSplit = -1;
    float invTotalArea = 1.0f / node.aabb.SurfaceArea();


    // Iterates over all possible split points. The loop goes from 0 to num_triangles - 2 
    // because splitting at the last triangle (num_triangles - 1) would not divide the node into two valid parts.
    for (int i = 0; i < num_triangles - 1; ++i) {
        float leftArea = leftBounds[i].SurfaceArea(); //leftBounds[i]: The cumulative AABB that contains all the triangles from the first triangle up to index i.
        float rightArea = rightBounds[i + 1].SurfaceArea(); //rightBounds[i + 1]: The cumulative AABB that contains all the triangles from index i + 1 to the last triangle.
        float cost = 0.125f + (leftArea * (i + 1) + rightArea * (num_triangles - i - 1)) * invTotalArea;

        if (cost < bestCost) {
            bestCost = cost;
            bestSplit = i;
        }
    }

    if (bestSplit == -1) {
        // If we can't find a valid split, create a leaf node
        largestLeafSize = num_triangles;
        return 1;
    }


    // Partition triangles based on the best split
    int left_triangle_count = bestSplit + 1;
    int right_triangle_count = num_triangles - left_triangle_count;

    int left_index = visited_node++;
    int right_index = visited_node++;

    node.leftChild = left_index;
    node.rightChild = right_index;
    nodes[left_index].fisrtT_idx = node.fisrtT_idx;
    nodes[left_index].TCount = left_triangle_count;
    nodes[right_index].fisrtT_idx = node.fisrtT_idx + left_triangle_count;
    nodes[right_index].TCount = right_triangle_count;
    node.TCount = 0;

    // Update the bounding boxes of the child nodes
    updateBVH(relations, nodes, left_index);
    updateBVH(relations, nodes, right_index);

    int largestLeafSize_L, largestLeafSize_R;
    int leftDepth = subdivide(relations, nodes, left_index, visited_node, largestLeafSize_L);
    int rightDepth = subdivide(relations, nodes, right_index, visited_node, largestLeafSize_R);

    largestLeafSize = glm::max(largestLeafSize_L, largestLeafSize_R);
    return glm::max(leftDepth, rightDepth) + 1;
}




