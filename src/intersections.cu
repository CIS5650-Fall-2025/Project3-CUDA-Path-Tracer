#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
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

__host__ __device__ float squareIntersectionTest(
    Geom square,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal)
{
    Ray q;
    q.origin = multiplyMV(square.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(square.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float t = -q.origin.z / q.direction.z;
    if (t <= 0)
    {
        return -1;
    }

    glm::vec3 point = getPointOnRay(q, t);
    if (abs(point.x) > 0.5 || abs(point.y) > 0.5)
    {
        return -1;
    }

    intersectionPoint = multiplyMV(square.transform, glm::vec4(point, 1.0f));
    normal = glm::normalize(multiplyMV(square.invTranspose, glm::vec4(0, 0, 1, 0)));
    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float doubleTriangleArea(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2) {
    return glm::length(glm::cross(p1 - p0, p2 - p0));
}

__host__ __device__ float triangleIntersectionTest(
    Tri tri,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal)
{
    normal = glm::normalize(glm::cross((tri.points[1] - tri.points[0]), tri.points[2] - tri.points[0]));
    glm::vec3 planePoint = tri.points[0];
    float t = (glm::dot(normal, planePoint) - glm::dot(normal, r.origin)) / glm::dot(normal, r.direction);
    if (t <= 0) {
        return -1;
    }
    intersectionPoint = getPointOnRay(r, t);

    glm::vec3 areas;
    for (int i = 0; i < 3; i++) {
        glm::mat3 points = tri.points;
        points[i] = intersectionPoint;
        areas[i] = doubleTriangleArea(points[0], points[1], points[2]);
    }
    float totalArea = doubleTriangleArea(tri.points[0], tri.points[1], tri.points[2]);
    if (totalArea < areas.x + areas.y + areas.z) {
        return -1;
    }
    return glm::length(intersectionPoint - r.origin) / glm::length(r.direction);
}

__device__ ShadeableIntersection queryIntersection(
    Ray ray,
    const Geom *geoms,
    int geomsSize,
    const Tri *tris,
    int trisSize)
{
    float t;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    float t_min = FLT_MAX;
    int hit_material = -1;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;

    for (int i = 0; i < geomsSize; i++)
    {
        const Geom &geom = geoms[i];

        if (geom.type == CUBE)
        {
            t = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == SPHERE)
        {
            t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == SQUARE)
        {
            t = squareIntersectionTest(geom, ray, tmp_intersect, tmp_normal);
        }

        if (t > 0.0f && t_min > t)
        {
            t_min = t;
            hit_material = geoms[i].materialid;
            intersect_point = tmp_intersect;
            normal = tmp_normal;
        }
    }

    for (int i = 0; i < trisSize; i++)
    {
        t = triangleIntersectionTest(tris[i], ray, tmp_intersect, tmp_normal);

        if (t > 0.0f && t_min > t)
        {
            t_min = t;
            hit_material = tris[i].materialid;
            intersect_point = tmp_intersect;
            normal = tmp_normal;
        }
    }

    ShadeableIntersection intersection;

    if (hit_material == -1)
    {
        intersection.t = -1.0f;
    }
    else
    {
        // The ray hits something
        intersection.t = t_min;
        intersection.materialId = hit_material;
        intersection.surfaceNormal = normal;
    }

    return intersection;
}

__device__ int queryIntersectionGeometryIndex(
    Ray ray,
    const Geom *geoms,
    int geomsSize,
    const Tri *tris,
    int trisSize)
{
    float t;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;

    for (int i = 0; i < geomsSize; i++)
    {
        const Geom &geom = geoms[i];

        if (geom.type == CUBE)
        {
            t = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == SPHERE)
        {
            t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == SQUARE)
        {
            t = squareIntersectionTest(geom, ray, tmp_intersect, tmp_normal);
        }

        if (t > 0.0f && t_min > t)
        {
            t_min = t;
            hit_geom_index = i;
        }
    }

    for (int i = 0; i < trisSize; i++)
    {
        t = triangleIntersectionTest(tris[i], ray, tmp_intersect, tmp_normal);

        if (t > 0.0f && t_min > t)
        {
            return -1;
        }
    }

    return hit_geom_index;
}