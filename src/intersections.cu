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
    int vertexSize,
    Vertex* vertices,
    bool toggleCulling)
{
    Ray rt;
    rt.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    rt.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    if (toggleCulling && !intersectBoundingBox(rt.origin, 1.0f / rt.direction, mesh.boundingBoxMin, mesh.boundingBoxMax, FLT_MAX)) return -1.0f;

    // Iterate through all triangles in this mesh
    float closestT = -1.0f;
    int bestTriIdx = -1;
    glm::vec2 v_range = mesh.vertex_indices;

    for (int i = v_range.x; i <= v_range.y; i += 3) {
        // Fetch triangle vertices
        glm::vec3 a = vertices[i].position;
        glm::vec3 b = vertices[i + 1].position;
        glm::vec3 c = vertices[i + 2].position;

        // Möller–Trumbore intersection
        glm::vec3 ab = b - a;
        glm::vec3 ac = c - a;
        glm::vec3 pvec = cross(r.direction, ac);
        float det = dot(ab, pvec);

        if (fabs(det) < EPSILON) continue;

        float invDet = 1.0f / det;
        glm::vec3 tvec = r.origin - a;
        float u = dot(tvec, pvec) * invDet;
        if (u < 0.0f || u > 1.0f) continue;

        glm::vec3 qvec = cross(tvec, ab);
        float v = dot(r.direction, qvec) * invDet;
        if (v < 0.0f || u + v > 1.0f) continue;

        float t = dot(ac, qvec) * invDet;
        if (t > EPSILON && (closestT == -1.0f || t < closestT)) {
            closestT = t;
            bestTriIdx = i;
        }
    }

    if (closestT < 0.0f) return -1.0f;

    intersectionPoint = r.origin + closestT * r.direction;
    glm::vec3 bary = barycentricInterp(
        vertices[bestTriIdx].position,
        vertices[bestTriIdx + 1].position,
        vertices[bestTriIdx + 2].position,
        intersectionPoint);

    // Interpolate normals
    normal =
        vertices[bestTriIdx].normal * bary.x +
        vertices[bestTriIdx + 1].normal * bary.y +
        vertices[bestTriIdx + 2].normal * bary.z;

    outside = dot(normalize(normal), normalize(r.direction)) <= 0.f;
    return closestT;
}

__host__ __device__ glm::vec3 barycentricInterp(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, const glm::vec3& p) {
    float areaABC = triArea(a, b, c);
    float areaPBC = triArea(p, b, c);
    float areaPCA = triArea(p, c, a);
    float areaPAB = triArea(p, a, b);
    return glm::vec3(areaPBC / areaABC, areaPCA / areaABC, areaPAB / areaABC);
}

__host__ __device__ float triArea(const glm::vec3& x, const glm::vec3& y, const glm::vec3& z) {
    return 0.5f * glm::length(glm::cross(z - y, x - y));
}
