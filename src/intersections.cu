#include "intersections.h"

__host__ __device__ float meshIntersectionTest(
    Geom mesh,
    Triangle* triangles,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    bool& outside) {

    Ray q;
    q.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = 1e38f;
    float tmax = 1e38f;

    float t = -1;
    glm::vec3 tmin_n;
    glm::vec2 tmin_uv;

    // algo referenced from 
    // https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-polygon-mesh/ray-tracing-polygon-mesh-part-1.html

    // loop through mesh triangles
    for (int i = 0; i < mesh.meshEnd; ++i) {

        glm::vec3 v0 = triangles[mesh.meshStart + i].verts[0];
        glm::vec3 v1 = triangles[mesh.meshStart + i].verts[1];
        glm::vec3 v2 = triangles[mesh.meshStart + i].verts[2];

        glm::vec3 baryPos;
        // check intersection with the current triangle
        if (glm::intersectRayTriangle(q.origin, q.direction, v0, v1, v2, baryPos)) {
            t = baryPos.z;  // intersection distance

            if (t > 0 && t < tmin) {
                tmin = t;

                // barycentric weights
                tmin_n = glm::normalize(
                    (1 - baryPos.x - baryPos.y) * triangles[mesh.meshStart + i].normals[0] +
                    baryPos.x * triangles[mesh.meshStart + i].normals[1] +
                    baryPos.y * triangles[mesh.meshStart + i].normals[2]);

                // use same weights to calculate uv coords
                if (mesh.hasTexture != -1) {
                    tmin_uv =
                        (1 - baryPos.x - baryPos.y) * triangles[mesh.meshStart + i].uvs[0] +
                        baryPos.x * triangles[mesh.meshStart + i].uvs[1] +
                        baryPos.y * triangles[mesh.meshStart + i].uvs[2];
                }

            }
        }
    }

    if (tmin < tmax && tmin > 0) {
        outside = true;

        uv = tmin_uv;

        intersectionPoint = multiplyMV(mesh.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    else {
        outside = false;
    }

    return -1;
}

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    Ray q;
    q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
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
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
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
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
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


__host__ __device__ float AABBIntersectionTest(const Ray& ray, const glm::vec3& bmin, const glm::vec3& bmax)
{
    // precompute inverse directions to replace divisions with multiplications
    glm::vec3 invDir;
    invDir.x = (ray.direction.x != 0.0f) ? 1.0f / ray.direction.x : 1e30f;
    invDir.y = (ray.direction.y != 0.0f) ? 1.0f / ray.direction.y : 1e30f;
    invDir.z = (ray.direction.z != 0.0f) ? 1.0f / ray.direction.z : 1e30f;

    // compute t values for each axis
    float tx1 = (bmin.x - ray.origin.x) * invDir.x;
    float tx2 = (bmax.x - ray.origin.x) * invDir.x;
    float tmin = glm::min(tx1, tx2);
    float tmax = glm::max(tx1, tx2);

    float ty1 = (bmin.y - ray.origin.y) * invDir.y;
    float ty2 = (bmax.y - ray.origin.y) * invDir.y;
    tmin = glm::max(tmin, glm::min(ty1, ty2));
    tmax = glm::min(tmax, glm::max(ty1, ty2));

    float tz1 = (bmin.z - ray.origin.z) * invDir.z;
    float tz2 = (bmax.z - ray.origin.z) * invDir.z;
    tmin = glm::max(tmin, glm::min(tz1, tz2));
    tmax = glm::min(tmax, glm::max(tz1, tz2));

    // intersection exists if tmax >= tmin and tmax > 0
    return (tmax >= tmin && tmax > 0.0f) ? tmin : 1e30f;
}

__host__ __device__ float BVHIntersectionTest(
    Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    BVHNode* bvhNode,
    Triangle* triangles,
    int& geomIdx,
    bool& outside)
{
    int stack[32];
    int stackPtr = 0;
    stack[stackPtr++] = 0; // root node at index 0

    float tmin = 1e38f;
    bool hasIntersection = false;

    while (stackPtr > 0)
    {
        // pop the top node from the stack
        int currentNodeIdx = stack[--stackPtr];
        BVHNode* node = &bvhNode[currentNodeIdx];

        if (node->triCount > 0) // is leaf node
        {
            for (int i = 0; i < node->triCount; i++)
            {
                Triangle triangle = triangles[node->leftFirst + i];

                glm::vec3 baryPos;
                // check intersection with the current triangle
                if (glm::intersectRayTriangle(r.origin, r.direction,
                    triangle.verts[0], triangle.verts[1], triangle.verts[2], baryPos))
                {
                    float t = baryPos.z;

                    if (t > 0.0f && t < tmin)
                    {
                        tmin = t;

                        intersectionPoint = (1.0f - baryPos.x - baryPos.y) * triangle.verts[0] +
                            baryPos.x * triangle.verts[1] + baryPos.y * triangle.verts[2];

                        normal = (1.0f - baryPos.x - baryPos.y) * triangle.normals[0] +
                            baryPos.x * triangle.normals[1] + baryPos.y * triangle.normals[2];

                        geomIdx = triangle.geomIdx;

                        uv = (1.0f - baryPos.x - baryPos.y) * triangle.uvs[0] +
                            baryPos.x * triangle.uvs[1] + baryPos.y * triangle.uvs[2];

                        outside = glm::dot(normal, r.direction) < 0.0f;
                        hasIntersection = true;
                    }
                }
            }
        }
        else // is internal node
        {
            // retrieve children and compute AABB intersection dists
            int leftFirstIdx = node->leftFirst;
            int rightChildIdx = node->leftFirst + 1;

            BVHNode* child1 = &bvhNode[leftFirstIdx];
            BVHNode* child2 = &bvhNode[rightChildIdx];

            float dist1 = AABBIntersectionTest(r, child1->aabbMin, child1->aabbMax);
            float dist2 = AABBIntersectionTest(r, child2->aabbMin, child2->aabbMax);

            // reorder children by dist
            if (dist1 > dist2)
            {
                std::swap(dist1, dist2);
                std::swap(child1, child2);
            }

            if (dist1 < tmin)
            {
                if (dist2 < tmin)
                {
                    // push into stack
                    stack[stackPtr++] = rightChildIdx;
                }
                stack[stackPtr++] = leftFirstIdx;
            }
        }
    }

    return hasIntersection ? tmin : -1.0f;
}

__host__ __device__ glm::vec2 hash(glm::vec2 uv)
{
    int n = uv[0] + uv[1] * 57;
    n = (n << 13) ^ n;
    float hf = 1.0 - float((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0;
    return glm::vec2(glm::fract(hf * 0.123456), glm::fract(hf * 0.654321));
}

__host__ __device__ float fbmWood(glm::vec2 uv) {

    // wood texture

    uv *= 10;

    float sum = 0.0f;
    int octaves = 10;
    float amp = 1.0f;
    float total = 0.f;
    float freq = 0.8f;

    for (int i = 1; i <= octaves; i++) {
        sum += glm::dot(hash(uv), hash(uv)) * amp;
        uv *= freq;
        total += amp;
    }
    return sum / total;
}

// referenced from my own 4600 shader homework 
__host__ __device__ float worleyMarble(glm::vec2 uv)
{
    // slight marbleing effect using worley noise

    float scale = 25.f;
    float dist;

    glm::vec2 uvInt = floor(uv * scale);
    glm::vec2 uvFract = glm::fract(uv * scale);

    float minDist = 10.f;

    // make cells
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            glm::vec2 neighbor = glm::vec2(x, y);
            glm::vec2 point = hash(neighbor + uvInt);
            glm::vec2 diff = neighbor + point - uvFract;
            dist = glm::dot(diff, diff) * 0.5;
            minDist = glm::min(minDist, dist);
        }
    }
    return 1.0 - minDist;
}

__host__ __device__ glm::vec3 generateProceduralTexture(
    ShadeableIntersection intersection
)
{
    if (intersection.textureId == -2) { // first procedural texture: wood

        float noise = fbmWood(intersection.uv);
        glm::vec3 outCol = glm::vec3(noise * 0.139, noise * 0.053, noise * 0.0014);
        return outCol;
    }
    else { // second procedural texture: marble 

        float worley = worleyMarble(intersection.uv);
        glm::vec3 outCol = glm::vec3(worley * 0.12, worley * 0.3, worley * 0.25);
        return outCol;
    }
}