#include "intersections.h"

__device__ void computeBarycentricWeights(const glm::vec3& p, const glm::vec3& A, const glm::vec3& B, const glm::vec3& C, glm::vec3& weights)
{
    // Calculate vectors

    glm::vec3 edgeAB = B - A;
    glm::vec3 edgeAC = C - A;
    glm::vec3 edgeAP = p - A;

    //Total Area (* 2)
    float S = length(cross(edgeAB, edgeAC));

    // Calculate areas of sub-triangles

    float areaPBC = length(cross(B - p,
        C - p));

    float areaPCA = length(cross(C - p, A - p));

    float areaPAB = length(cross(A - p, B - p));

    // Calculate barycentric coordinates
    weights[0] = areaPBC / S;
    weights[1] = areaPCA / S;
    weights[2] = areaPAB / S;
}

__host__ __device__ float triangleIntersectionTest(
    Ray r,
    const MeshTriangle& tri,
    glm::vec3& intersectionPoint,
    glm::vec3& normal)
{
    float t = -1;

    glm::vec3 edge1 = tri.v1 - tri.v0;
    glm::vec3 edge2 = tri.v2 - tri.v0;

    normal = glm::normalize(glm::cross(edge1, edge2));

    // Compute the det
    glm::vec3 pvec = glm::cross(r.direction, edge2);
    float det = glm::dot(edge1, pvec);

    // If determinant is near zero, ray is parallel
    if (fabs(det) < EPSILON) return -1;

    float invDet = 1.0f / det;

    glm::vec3 tvec = r.origin - tri.v0;

    float u = glm::dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return -1;

    glm::vec3 qvec = glm::cross(tvec, edge1);
    float v = glm::dot(r.direction, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return -1;

    // Calculate t, the intersection distance along the ray
    t = glm::dot(edge2, qvec) * invDet;

    // If t is negative, the intersection is behind the ray origin (no hit)
    if (t < EPSILON) return -1;

    intersectionPoint = getPointOnRay(r, t);
     
    return t;
}

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = glm::min(t1, t2);
        outside = true;
    }
    else
    {
        t = glm::max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__device__ bool intersectAABB(const Ray& r, const AABB& aabb) {
    float tMin = (aabb.min.x - r.origin.x) / r.direction.x;
    float tMax = (aabb.max.x - r.origin.x) / r.direction.x;

    if (tMin > tMax) {
        // Swap if tMin > tMax
        float temp = tMin;
        tMin = tMax;
        tMax = temp;
    }

    float tyMin = (aabb.min.y - r.origin.y) / r.direction.y;
    float tyMax = (aabb.max.y - r.origin.y) / r.direction.y;

    if (tyMin > tyMax) {
        // Swap if tyMin > tyMax
        float temp = tyMin;
        tyMin = tyMax;
        tyMax = temp;
    }

    if ((tMin > tyMax) || (tyMin > tMax)) {
        // No intersection
        return false;
    }

    // Calculate the intersection intervals
    if (tyMin > tMin) {
        tMin = tyMin;  // Update tMin
    }
    if (tyMax < tMax) {
        tMax = tyMax;  // Update tMax
    }

    // Repeat for the Z axis
    float tzMin = (aabb.min.z - r.origin.z) / r.direction.z;
    float tzMax = (aabb.max.z - r.origin.z) / r.direction.z;

    if (tzMin > tzMax) {
        // Swap if tzMin > tzMax
        float temp = tzMin;
        tzMin = tzMax;
        tzMax = temp;
    }

    // Check for intersection
    if ((tMin > tzMax) || (tzMin > tMax)) {
        // No intersection
        return false;
    }

    return true;  // The ray intersects the bounding box
}

__device__ void BVHIntersect(Ray r, ShadeableIntersection& intersection,
    MeshTriangle* triangles, BVHNode* bvhNodes, cudaTextureObject_t* texObjs)
{
    float t;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    glm::vec3 texCol;
    int matId = 0;

    int stack[16];
    int stackPtr = 0;
    stack[stackPtr] = 0;
    stackPtr++;

    while (stackPtr > 0) {
        if (stackPtr >= 16) {
            // Stack overflow, exit traversal
            return;
        }

        int nodeIdx = stack[--stackPtr];

        if (nodeIdx < 0) {
            continue;
        }

        const BVHNode& node = bvhNodes[nodeIdx];

        //IF LEAF
        if (node.triangleIDs.x != -1) {
            for (int j = 0; j < 4; j++) {
                int tri_idx = node.triangleIDs[j];
                if (tri_idx != -1) {
                    const MeshTriangle& tri = triangles[tri_idx];
                    glm::vec3 tmp_intersect;
                    glm::vec3 tmp_normal;
                    glm::vec3 tmp_texCol = glm::vec3(-1, -1, -1);
                    int tmp_matId = tri.materialIndex;

                    t = triangleIntersectionTest(r, tri, tmp_intersect, tmp_normal);

                    if (t > 0.0f && t_min > t)
                    {
                        t_min = t;
                        hit_geom_index = 1;
                        intersect_point = tmp_intersect;
                        matId = tmp_matId;

                        if (tri.baseColorTexID != -1) {
                            cudaTextureObject_t texObj = texObjs[tri.baseColorTexID];
                            glm::vec2 UV = glm::vec2(0.5f, 0.5f);

                            glm::vec3 weights;
                            computeBarycentricWeights(intersect_point, tri.v0,
                                tri.v1,
                                tri.v2,
                                weights);

                            UV = weights.x * tri.uv0 +
                                weights.y * tri.uv1 +
                                weights.z * tri.uv2;
                            bool isInt = true;
                            if (isInt) {
                                int4 texColor_flt = tex2D<int4>(texObj, UV.x, UV.y);
                                tmp_texCol = glm::vec3(texColor_flt.x / 255.f, texColor_flt.y / 255.f, texColor_flt.z / 255.f);
                            }
                            else {
                                float4 texColor_flt = tex2D<float4>(texObj, UV.x, UV.y);
                                tmp_texCol = glm::vec3(texColor_flt.x, texColor_flt.y, texColor_flt.z);
                            }
                            tmp_texCol = glm::max(tmp_texCol, glm::vec3(EPSILON));
                        }

                        if (tri.normalMapTexID != -1) {
                            cudaTextureObject_t texObj = texObjs[tri.normalMapTexID];
                            glm::vec2 UV = glm::vec2(0.5f, 0.5f);

                            glm::vec3 weights;
                            computeBarycentricWeights(intersect_point, tri.v0,
                                tri.v1,
                                tri.v2,
                                weights);

                            UV = weights.x * tri.uv0 +
                                weights.y * tri.uv1 +
                                weights.z * tri.uv2;
                            bool isInt = true;
                            if (isInt) {
                                int4 normalEncoded = tex2D<int4>(texObj, UV.x, UV.y);
                                tmp_normal = glm::vec3(normalEncoded.x / 255.f, normalEncoded.y / 255.f, normalEncoded.z / 255.f);
                                tmp_normal = (tmp_normal * 2.f) - glm::vec3(1.f);
                            }
                            else {
                                float4 normalEncoded = tex2D<float4>(texObj, UV.x, UV.y);
                                tmp_normal = glm::vec3(normalEncoded.x, normalEncoded.y, normalEncoded.z);
                                tmp_normal = (tmp_normal * 2.f) - glm::vec3(1.f);
                            }
                            tmp_normal = normalize(tmp_normal); //IMPORTANT
                        }
                        //This code here is for both textured and NON textured!
                        texCol = tmp_texCol;
                        normal = tmp_normal;
                    }
                }
                else {
                    break;
                }
            }
        }
        else {
            //IF NOT LEAF
            int leftIdx = node.leftChild;
            int rightIdx = node.rightChild;

            bool hitLeft = intersectAABB(r, bvhNodes[leftIdx].bounds);
            bool hitRight = intersectAABB(r, bvhNodes[rightIdx].bounds);

            if (hitLeft) stack[stackPtr++] = leftIdx;
            if (hitRight) stack[stackPtr++] = rightIdx;
        }
    }
    if (hit_geom_index == -1)
    {
        intersection.t = -1.0f;
        intersection.materialId = -1;
    }
    else
    {
        intersection.t = t_min;
        intersection.materialId = matId;
        intersection.surfaceNormal = normal;
        intersection.texCol = texCol;
    }
}