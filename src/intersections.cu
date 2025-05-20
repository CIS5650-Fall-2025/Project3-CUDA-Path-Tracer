#include "intersections.h"

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
        t = std::min(t1, t2);
        outside = true;
    }
    else
    {
        t = std::max(t1, t2);
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


__host__ __device__ bool intersectBoundingBox(glm::vec3 origin, glm::vec3 invDir, glm::vec3 boxMin, glm::vec3 boxMax, float t) {
    float tx1 = (boxMin.x - origin.x) * invDir.x;
    float tx2 = (boxMax.x - origin.x) * invDir.x;

    double tmin = std::min(tx1, tx2);
    double tmax = std::max(tx1, tx2);

    float ty1 = (boxMin.y - origin.y) * invDir.y;
    float ty2 = (boxMax.y - origin.y) * invDir.y;

    tmin = std::max(tmin, static_cast<double>(std::min(ty1, ty2)));
    tmax = std::min(tmax, static_cast<double>(std::max(ty1, ty2)));

    float tz1 = (boxMin.z - origin.z) * invDir.z;
    float tz2 = (boxMax.z - origin.z) * invDir.z;

    tmin = std::max(tmin, static_cast<double>(std::min(tz1, tz2)));
    tmax = std::min(tmax, static_cast<double>(std::max(tz1, tz2)));

    return tmax >= std::max(static_cast<double>(0.0f), tmin) && tmin < t;
}


__host__ __device__ float meshRayIntersectionTest(
    Geom mesh,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    Vertex* vertices,
    bool toggleCulling)
{
    Ray rm;
    rm.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    rm.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    if (toggleCulling && !intersectBoundingBox(rm.origin, 1.0f / rm.direction, mesh.boundingBoxMin, mesh.boundingBoxMax, FLT_MAX)) return -1.0f;
    float tmin = 1e38f;
    int bestTriIdx = -1;
    glm::vec3 hitNormal;
    // Indices for this mesh in global vertex list
    int startIdx = static_cast<int>(mesh.vertex_indices.x);
    int endIdx = static_cast<int>(mesh.vertex_indices.y);

    // Loop through all triangles belonging to this mesh
    for (int vIdx = startIdx; vIdx <= endIdx; vIdx += 3) {
        // Fetch triangle vertex positions
        glm::vec3 v0 = vertices[vIdx].position;
        glm::vec3 v1 = vertices[vIdx + 1].position;
        glm::vec3 v2 = vertices[vIdx + 2].position;
        glm::vec3 baryCoord;

        // Use GLM's built-in ray-triangle intersection
        if (glm::intersectRayTriangle(rm.origin, rm.direction, v0, v1, v2, baryCoord)) {
            float t_hit = baryCoord.z;
            if (t_hit > 0 && t_hit < tmin) {
                tmin = t_hit;
                bestTriIdx = vIdx;
                hitNormal = baryCoord;
            }
        }
    }
    if (bestTriIdx >= 0) {
        // Found intersection: interpolate world-space position and normal
        glm::vec3 v0 = vertices[bestTriIdx].position;
        glm::vec3 v1 = vertices[bestTriIdx + 1].position;
        glm::vec3 v2 = vertices[bestTriIdx + 2].position;
        glm::vec3 n0 = vertices[bestTriIdx].normal;
        glm::vec3 n1 = vertices[bestTriIdx + 1].normal;
        glm::vec3 n2 = vertices[bestTriIdx + 2].normal;
        glm::vec3 localIntersect = rm.origin + tmin * rm.direction;
        intersectionPoint = multiplyMV(mesh.transform, glm::vec4(localIntersect, 1.0f));
        glm::vec3 interpolatedNormal = glm::normalize(
            (1 - hitNormal.x - hitNormal.y) * n0 +
            hitNormal.x * n1 +
            hitNormal.y * n2
        );
        normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(interpolatedNormal, 0.0f)));

        // "outside" if normal faces against ray direction
        outside = glm::dot(glm::normalize(normal), glm::normalize(r.direction)) < 0.f;

        return glm::length(r.origin - intersectionPoint);
    }
    else {
        outside = false;
        return -1.0f;
    }
}