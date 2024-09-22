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

// implement the kernal function that performs the Moller-Trumbore ray-triangle intersection test
__host__ __device__ float intersect(const Ray ray,
                                    const glm::vec3 point_0,
                                    const glm::vec3 point_1,
                                    const glm::vec3 point_2) {

    // declare the variables for the two edges of the triangle and other vector values
    glm::vec3 edge_1, edge_2, h, s, q;

    // declare the variables of the intermediate scalars for calculation
    float a, f, u, v, t;

    // compute the edge vectors
    edge_1 = point_1 - point_0;
    edge_2 = point_2 - point_0;

    // compute the normal vector formed by the ray and an edge
    h = glm::cross(ray.direction, edge_2);

    // compute the cosine value of the angle between the normal vector and the other edge
    a = glm::dot(edge_1, h);

    // return a miss when the ray is parallel to the triangle, suggested by the cosine value
    if (-EPSILON < a && a < EPSILON) {
        return FLT_MAX;
    }

    // compute the reciprocal of a for scaling
    f = 1.0f / a;

    // compute the vector from the first vertex to the ray's origin
    s = ray.origin - point_0;

    // calculate one of the barycentric coordinate
    u = f * glm::dot(s, h);

    // return a miss when the intersection point is outside the triangle, suggested by u
    if (u < 0.0f || 1.0f < u) {
        return FLT_MAX;
    }

    // compute the normal vector formed by s and one of the edges
    q = glm::cross(s, edge_1);

    // calculate the other barycentric coordinate
    v = f * glm::dot(ray.direction, q);

    // return a miss when the intersection point is outside the triangle, suggested by u and v
    if (v < 0.0f || 1.0f < u + v) {
        return FLT_MAX;
    }

    // compute the intersection distance
    t = f * dot(edge_2, q);

    // return the distance when it is positive
    if (t > EPSILON) {
        return t;

        // return a miss otherwise
    } else {
        return FLT_MAX;
    }
}

// implement the kernal function that computes the barycentric weights
__host__ __device__ glm::vec3 compute(const glm::vec3 intersection_point,
                                      const glm::vec3 point_0,
                                      const glm::vec3 point_1,
                                      const glm::vec3 point_2) {

    // declare the variables for the two edges of the triangle
    glm::vec3 edge_1, edge_2;

    // declare the scalar variables of the intermediate triangle sizes
    float size, size_0, size_1, size_2;

    // compute the edge vectors
    edge_1 = point_1 - point_0;
    edge_2 = point_2 - point_1;

    // compute the size of the triangle
    size = glm::length(glm::cross(edge_1, edge_2));

    // compute the new edge vectors
    edge_1 = intersection_point - point_1;
    edge_2 = intersection_point - point_2;

    // compute the size of one of the sub triangles
    size_0 = glm::length(glm::cross(edge_1, edge_2));

    // compute the new edge vectors
    edge_1 = intersection_point - point_0;
    edge_2 = intersection_point - point_2;

    // compute the size of one of the sub triangles
    size_1 = glm::length(glm::cross(edge_1, edge_2));

    // compute the new edge vectors
    edge_1 = intersection_point - point_0;
    edge_2 = intersection_point - point_1;

    // compute the size of one of the sub triangles
    size_2 = glm::length(glm::cross(edge_1, edge_2));

    // return the barycentric weights
    return glm::vec3(size_0, size_1, size_2) / size;
}
