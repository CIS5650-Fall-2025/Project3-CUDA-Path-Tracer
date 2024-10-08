#include "intersections.h"

__host__ __device__ void solveQuadratic(float A, float B, float C, float& t0, float& t1)
{
    float invA = 1.0 / A;
    B *= invA;
    C *= invA;
    float neg_halfB = -B * 0.5;
    float u2 = neg_halfB * neg_halfB - C;
    float u = u2 < 0.0 ? neg_halfB = 0.0 : sqrt(u2);
    t0 = neg_halfB - u;
    t1 = neg_halfB + u;
}

__host__ __device__ float boxIntersectionTest(
    const Geom& box,
    const Ray& r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.transforms.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.transforms.inverseTransform, glm::vec4(r.direction, 0.0f)));

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
        intersectionPoint = multiplyMV(box.transforms.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.transforms.invTranspose, glm::vec4(tmin_n, 0.0f)));
        if (!outside) normal = -normal;
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float boxIntersectionTest(const Geom& box, const Ray& r)
{
    Ray q;
    q.origin = multiplyMV(box.transforms.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(box.transforms.inverseTransform, glm::vec4(r.direction, 0.0f)));

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
        if (tmin <= 0.f)
        {
            tmin = tmax;
        }
        glm::vec3 intersectionPoint = multiplyMV(box.transforms.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    const Geom& sphere,
    const Ray& r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.transforms.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.transforms.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = glm::sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0.f && t2 < 0.f) {
        return -1;
    }
    else if (t1 > 0.f && t2 > 0.f) {
        t = glm::min(t1, t2);
    }
    else {
        t = glm::max(t1, t2);
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transforms.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.transforms.invTranspose, glm::vec4(objspaceIntersection, 0.f)));

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float sphereIntersectionTest(const Geom& sphere, const Ray& r)
{
    Ray ray;
    float radius = 0.5f;
    ray.origin = glm::vec3(sphere.transforms.inverseTransform * glm::vec4(r.origin, 1.));
    ray.direction = glm::vec3(sphere.transforms.inverseTransform * glm::vec4(r.direction, 0.));
    float t0, t1;
    glm::vec3 diff = ray.origin;
    float a = glm::dot(ray.direction, ray.direction);
    float b = 2.f * glm::dot(ray.direction, diff);
    float c = glm::dot(diff, diff) - (radius * radius);
    solveQuadratic(a, b, c, t0, t1);
    glm::vec3 localNor = t0 > 0.f ? ray.origin + t0 * ray.direction : ray.origin + t1 * ray.direction;
    localNor = glm::normalize(localNor);
    return t0 > 0.f ? t0 : t1 > 0.f ? t1 : -1;
    /*
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.transforms.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.transforms.inverseTransform, glm::vec4(r.direction, 0.0f)));

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
    }
    else
    {
        t = glm::max(t1, t2);
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    glm::vec3 intersectionPoint = multiplyMV(sphere.transforms.transform, glm::vec4(objspaceIntersection, 1.f));

    return glm::length(r.origin - intersectionPoint);
    */
}

__host__ __device__ float triangleIntersection(const Ray& r,
    const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& p3,
    glm::vec3& normal, glm::vec3& bary)
{
    glm::vec3 e1 = p2 - p1;
    glm::vec3 e2 = p3 - p1;
    glm::vec3 s = r.origin - p1;
    glm::vec3 s1 = glm::cross(r.direction, e2);
    glm::vec3 s2 = glm::cross(s, e1);
    float dnmt = glm::dot(s1, e1);

    if (glm::abs(dnmt) < EPSILON) return -1.f;
    dnmt = 1.f / dnmt;

    float t = glm::dot(s2, e2) * dnmt;
    if (t < 0.f) return -1.f;

    float b1 = glm::dot(s1, s) * dnmt;
    float b2 = glm::dot(s2, r.direction) * dnmt;

    if (b1 >= 0.f && b2 >= 0.f && (b1 + b2) <= 1.f) {
        normal = glm::normalize(glm::cross(e1, e2));
        bary = glm::vec3(1.f - b1 - b2, b1, b2);
        return t;
    }
    else {
        return -1.f;
    }
}

__host__ __device__ float triangleIntersectionTest(const Ray& r,
    const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& p3)
{
    glm::vec3 e1 = p2 - p1;
    glm::vec3 e2 = p3 - p1;
    glm::vec3 s = r.origin - p1;
    glm::vec3 s1 = glm::cross(r.direction, e2);
    glm::vec3 s2 = glm::cross(s, e1);
    float dnmt = glm::dot(s1, e1);

    if (glm::abs(dnmt) < EPSILON) return -1.f;
    dnmt = 1.f / dnmt;

    float t = glm::dot(s2, e2) * dnmt;
    if (t < 0.f) return -1.f;

    float b1 = glm::dot(s1, s) * dnmt;
    float b2 = glm::dot(s2, r.direction) * dnmt;

    if (b1 >= 0.f && b2 >= 0.f && (b1 + b2) <= 1.f)
        return t;
    else
        return -1.f;
}
