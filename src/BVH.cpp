#include "BVH.h"

////////////////////////////////// AABB Struct //////////////////////////////////
//                                                                             //
//                                                                             //
AABB calculateAABB(const Geom& geom)
{
    if (geom.type == TRIANGLE)
    {
        return calculateAABBTriangle(geom);
    }
    else if (geom.type == CUBE)
    {
        return calculateAABBCube(geom);
    }
    else
    {
        return calculateAABBSphere(geom);
    }
}

AABB calculateAABBTriangle(const Geom& geom)
{
    AABB aabb;
    aabb.expand(geom.v0);
    aabb.expand(geom.v1);
    aabb.expand(geom.v2);
    return aabb;
}

AABB calculateAABBCube(const Geom& cube)
{
    AABB aabb;

    // 1: Define the cube's unscaled half-size
    float halfSize = 0.5f;

    // 2: Define object space cube AABB corners
    glm::vec3 corners[8] = {
        { -halfSize, -halfSize, -halfSize },
        { -halfSize, -halfSize,  halfSize },
        { -halfSize,  halfSize, -halfSize },
        { -halfSize,  halfSize,  halfSize },
        {  halfSize, -halfSize, -halfSize },
        {  halfSize, -halfSize,  halfSize },
        {  halfSize,  halfSize, -halfSize },
        {  halfSize,  halfSize,  halfSize }
    };

    // 3: Transform each corner to world space and expand the AABB
    for (int i = 0; i < 8; ++i)
    {
        glm::vec3 worldPos = glm::vec3(cube.transform * glm::vec4(corners[i], 1.0f));
        aabb.expand(worldPos);
    }

    // 4: Return the computed AABB
    return aabb;
}

AABB calculateAABBSphere(const Geom& sphere)
{
    AABB aabb;

    // 1: Define the sphere's unscaled radius
    float radius = 0.5f;

    // 2: Define object space sphere AABB corners
    glm::vec3 corners[8] = {
        { -radius, -radius, -radius },
        { -radius, -radius,  radius },
        { -radius,  radius, -radius },
        { -radius,  radius,  radius },
        {  radius, -radius, -radius },
        {  radius, -radius,  radius },
        {  radius,  radius, -radius },
        {  radius,  radius,  radius }
    };

    // 3: Transform each corner to world space and expand the AABB
    for (int i = 0; i < 8; ++i)
    {
        glm::vec3 worldPos = glm::vec3(sphere.transform * glm::vec4(corners[i], 1.0f));
        aabb.expand(worldPos);
    }

    // 4: Return the computed AABB
    return aabb;
}

////////////////////////////////// BVHNode Struct //////////////////////////////////
//                                                                                //
//                                                                                //
glm::vec3 calculateCentroid(const Geom& geom)
{
    if (geom.type == TRIANGLE)
    {
        return calculateCentroidTriangle(geom);
    }
    else if (geom.type == CUBE)
    {
        return calculateCentroidCube(geom);
    }
    else
    {
        return calculateCentroidSphere(geom);
    }
}

glm::vec3 calculateCentroidTriangle(const Geom& geom)
{
    return (geom.v0 + geom.v1 + geom.v2) / 3.0f;
}

glm::vec3 calculateCentroidCube(const Geom& cube)
{
    // Object space centroid of the cube (0, 0, 0)
    glm::vec4 objectSpaceCentroid(0.0f, 0.0f, 0.0f, 1.0f);

    // Transform the centroid to world space
    glm::vec4 worldSpaceCentroid = cube.transform * objectSpaceCentroid;

    // Return the result as a glm::vec3 (discard the w component)
    return glm::vec3(worldSpaceCentroid);
}

glm::vec3 calculateCentroidSphere(const Geom& sphere)
{
    // Object space centroid of the sphere (0, 0, 0)
    glm::vec4 objectSpaceCentroid(0.0f, 0.0f, 0.0f, 1.0f);

    // Transform the centroid to world space
    glm::vec4 worldSpaceCentroid = sphere.transform * objectSpaceCentroid;

    // Return the result as a glm::vec3 (discard the w component)
    return glm::vec3(worldSpaceCentroid);
}

////////////////////////////////// BVHBuilder Class //////////////////////////////////
//                                                                                  //
//                                                                                  //
BVHBuilder::BVHBuilder(std::vector<Geom>& _geoms, std::vector<BVHNode>& _bvhNodes)
{
    geoms = &_geoms;
    bvhNodes = &_bvhNodes;
}

BVHBuilder::~BVHBuilder()
{
    geoms = nullptr;
    bvhNodes = nullptr;
}

void BVHBuilder::initBVH()
{
    geomIdx.clear();
    geomIdx.resize(geoms->size());
    // Fill geomIdx with indices 0, 1, 2, ..., n-1
    std::iota(geomIdx.begin(), geomIdx.end(), 0);

    bvhNodes->clear();
    bvhNodes->resize(geoms->size() * 4);
    

    rootNodeIdx = 0;
    nodesUsed = 1;
}

int BVHBuilder::buildBVH(std::vector<int>& _geomIdx)
{
    initBVH();

    // assign all Geoms to root node
    BVHNode& root = (*bvhNodes)[rootNodeIdx];
    root.leftFirst = 0;
    root.geomCount = geoms->size();

    UpdateNodeBounds(rootNodeIdx);

    // Recursive subdivide
    Subdivide(rootNodeIdx);

    bvhNodes->erase(
        std::remove_if(
            bvhNodes->begin(),
            bvhNodes->end(),
            [](const BVHNode& node) { return node.leftFirst == 0 && node.geomCount == 0; }
        ),
        bvhNodes->end()
    );

    _geomIdx = geomIdx;
    return rootNodeIdx;
}

void BVHBuilder::UpdateNodeBounds(int nodeIdx)
{
    BVHNode& node = (*bvhNodes)[nodeIdx];
    int first = node.leftFirst;

    for (int i = 0; i < node.geomCount; ++i)
    {
        AABB geomAABB = calculateAABB((*geoms)[geomIdx[first + i]]);
        node.aabb.expand(geomAABB);
    }
}

void BVHBuilder::Subdivide(int nodeIdx)
{
    // terminate recursion
    BVHNode& node = (*bvhNodes)[nodeIdx];
    //if (node.geomCount <= 2) return;

    // determine split axis and position
    int axis;
    float splitPos;
    //node.aabb.getSplitPlaneAxisAndPosition_ByLongestAxis(axis, splitPos);
    float bestCost = getSplitPlaneAxisAndPosition_SurfaceAreaHueristic(axis, splitPos, node);
    float parentSurfaceArea = node.aabb.getSurfaceArea();
    float parentCost = node.geomCount * parentSurfaceArea;
    if (bestCost >= parentCost) return;

    // in-place partition
    int i = node.leftFirst;
    int j = i + node.geomCount - 1;
    while (i <= j)
    {
        if (calculateCentroid((*geoms)[geomIdx[i]])[axis] < splitPos)
            i++;
        else
            std::swap(geomIdx[i], geomIdx[j--]);
    }
    // abort split if one of the sides is empty
    int leftCount = i - node.leftFirst;
    if (leftCount == 0 || leftCount == node.geomCount) return;
    // create child nodes
    int leftChildIdx = nodesUsed++;
    int rightChildIdx = nodesUsed++;
    (*bvhNodes)[leftChildIdx].leftFirst = node.leftFirst;
    (*bvhNodes)[leftChildIdx].geomCount = leftCount;
    (*bvhNodes)[rightChildIdx].leftFirst = i;
    (*bvhNodes)[rightChildIdx].geomCount = node.geomCount - leftCount;
    node.leftFirst = leftChildIdx;
    node.geomCount = 0;
    UpdateNodeBounds(leftChildIdx);
    UpdateNodeBounds(rightChildIdx);
    // recurse
    Subdivide(leftChildIdx);
    Subdivide(rightChildIdx);
}

int BVHBuilder::getSplitPlaneAxisAndPosition_SurfaceAreaHueristic(int& splitAxis, float& splitPosition, const BVHNode& node)
{
    int bestAxis = -1;
    float bestPos = 0;
    float bestCost = FLT_MAX;
    for (int axis = 0; axis < 3; ++axis) {
        for (int i = 0; i < node.geomCount; ++i) {
            const Geom& geom = (*geoms)[geomIdx[node.leftFirst + i]];
            float candidatePos = calculateCentroid(geom)[axis];

            AABB leftbox, rightbox;
            int leftCount = 0;
            int rightCount = 0;
            for (int j = 0; j < node.geomCount; ++j) {
                const Geom& _geom = (*geoms)[geomIdx[node.leftFirst + j]];
                if (calculateCentroid(_geom)[axis] < candidatePos) {
                    leftCount++;
                    leftbox.expand(calculateAABB(_geom));
                }
                else {
                    rightCount++;
                    rightbox.expand(calculateAABB(_geom));
                }
            }
            float leftCost = leftCount > 0 ? leftCount * leftbox.getSurfaceArea() : 0.0f;
            float rightCost = rightCount > 0 ? rightCount * rightbox.getSurfaceArea() : 0.0f;
            float cost = leftCost + rightCost;
            cost = cost >= FLT_EPSILON ? cost : FLT_MAX;

            if (cost < bestCost) {
                bestPos = candidatePos;
                bestAxis = axis;
                bestCost = cost;
            }
        }
    }
    splitAxis = bestAxis;
    splitPosition = bestPos;
    return bestCost;
}