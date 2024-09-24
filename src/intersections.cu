#include "intersections.h"

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

__host__ __device__ float triangleIntersectionTest(
    Geom geom_triangle,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    Triangle* dev_tris,
    int tri_index) 
{

    glm::vec3 edge1, edge2, h, s, q;
    float a, f, u, v;

    auto triangle = dev_tris[tri_index];

    //multiply ray by inv transform to make it an easy intersection test?
    glm::vec3 ro = multiplyMV(geom_triangle.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(geom_triangle.inverseTransform, glm::vec4(r.direction, 0.0f)));

    glm::vec3 p0 = triangle.v0.pos;
    glm::vec3 p1 = triangle.v1.pos;
    glm::vec3 p2 = triangle.v2.pos;

    edge1 = p1 - p0;
    edge2 = p2 - p0;
    h = glm::cross(rd, edge2);
    a = glm::dot(edge1, h);
    if (a > -EPSILON && a < EPSILON) {
        outside = true;
        return INFINITY;    // This ray is parallel to this triangle.
    }
    f = 1.0f / a;
    s = ro - p0;
    u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) {
        outside = true;
        return INFINITY;
    }
    q = glm::cross(s, edge1);
    v = f * glm::dot(rd, q);
    if (v < 0.0 || u + v > 1.0) {
        outside = true;
        return INFINITY;
    }
    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = f * dot(edge2, q);
    if (t > EPSILON) {
        outside = false;

        glm::vec3 objspaceIntersection = getPointOnRay(r, t);

        intersectionPoint = multiplyMV(geom_triangle.transform, glm::vec4(objspaceIntersection, 1.f));
        normal = glm::normalize(multiplyMV(geom_triangle.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
        return t;
    }
    else {
        // This means that there is a line intersection but not a ray intersection.
        outside = true;
        return INFINITY;
    }
    return INFINITY;
}

__host__ __device__ float meshIntersectionTest(
    Geom mesh,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    Triangle* dev_tris,
    int num_tris)
{
    float min_t = INFINITY;

    //multiply ray by inv transform to make it an easy intersection test
    glm::vec3 ro = r.origin;
    glm::vec3 rd = r.direction;

    for (int i = 0; i < num_tris; i++) {
        Triangle& triangle = dev_tris[i];
        glm::vec3 p0 = triangle.v0.pos;
        glm::vec3 p1 = triangle.v1.pos;
        glm::vec3 p2 = triangle.v2.pos;
        glm::vec3 out_pos;

        glm::vec3 edge1 = p1 - p0;
        glm::vec3 edge2 = p2 - p0;
        glm::vec3 h = glm::cross(rd, edge2);
        float a = glm::dot(edge1, h);
        if (a > -EPSILON && a < EPSILON) {
            continue;    // This ray is parallel to this triangle.
        }
        float f = 1.0f / a;
        glm::vec3 s = ro - p0;
        float u = f * glm::dot(s, h);
        if (u < 0.0 || u > 1.0) {
            continue;
        }
        glm::vec3 q = glm::cross(s, edge1);
        float v = f * glm::dot(rd, q);
        if (v < 0.0 || u + v > 1.0) {
            continue;
        }
        // At this stage we can compute t to find out where the intersection point is on the line.
        float t = f * dot(edge2, q);
        if (t > EPSILON) {
            outside = false;

            glm::vec3 objspaceIntersection = getPointOnRay(r, t);

            if (t < min_t) {
                //intersectionPoint = multiplyMV(mesh.transform, glm::vec4(objspaceIntersection, 1.f));
                //normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
                intersectionPoint = objspaceIntersection;;
                normal = triangle.v0.nor;
                min_t = t;
            }
        }
        else {
            // This means that there is a line intersection but not a ray intersection.
            continue;
        }
    }
    return min_t;
}