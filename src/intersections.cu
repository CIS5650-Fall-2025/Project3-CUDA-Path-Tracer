#include "intersections.h"

__host__ __device__ void computeBarycentricWeights(const glm::vec3& p, const glm::vec3& A, const glm::vec3& B, const glm::vec3& C, glm::vec3& weights)
{
    // Calculate vectors
    //float3 AB = make_float3(B.x - v0.x, B.y - v0.y, B.z - v0.z);
    //float3 AC = make_float3(C.x - v0.x, C.y - v0.y, C.z - v0.z);
    //float3 AP = make_float3(P.x - v0.x, P.y - A.y, P.z - A.z);

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
    Geom sphere,
    Ray r,
    const MeshTriangle& tri,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    float t = -1;  // Initialize to no intersection

    glm::vec3 edge1 = tri.v1- tri.v0;
    glm::vec3 edge2 = tri.v2 - tri.v0;

    glm::vec3 h = glm::cross(r.direction, edge2);
    float a = glm::dot(edge1, h);

    if (a > -EPSILON && a < EPSILON)
        return -1;  // Parallel case!

    float f = 1.0f / a;
    glm::vec3 s = r.origin - tri.v0;
    float u = f * glm::dot(s, h);

    if (u < 0.0f || u > 1.0f)
        return -1;

    glm::vec3 q = glm::cross(s, edge1);
    float v = f * dot(r.direction, q);

    if (v < 0.0f || u + v > 1.0f)
        return -1;

    t = f * dot(edge2, q);

    if (t > EPSILON) {
        normal = normalize(glm::cross(edge1, edge2));
        intersectionPoint = getPointOnRay(r, t);
        return t;
    }

    return -1;  // No intersection
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
