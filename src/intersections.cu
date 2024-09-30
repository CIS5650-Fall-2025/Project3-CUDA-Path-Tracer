#include "intersections.h"

#include "samplers.h"

__host__ __device__ float bboxIntersectionTest(
    BBox bbox,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside,
    glm::vec2 &times)
{
    glm::vec3 o = r.origin;
    glm::vec3 d = r.direction;

    glm::vec3 invdir = glm::vec3(1.f / r.direction.x, 1.f / r.direction.y, 1.f / r.direction.z);

    float t0x, t0y, t0z, t1x, t1y, t1z;

    glm::vec3 min = bbox.min;
    glm::vec3 max = bbox.max;

    // check if 2D rays are outside bbox
    if(invdir.x >= 0) {
        t0x = (min.x - o.x) * invdir.x;
        t1x = (max.x - o.x) * invdir.x;
    } else {
        t0x = (max.x - o.x) * invdir.x;
        t1x = (min.x - o.x) * invdir.x;
    }
    if(invdir.y >= 0) {
        t0y = (min.y - o.y) * invdir.y;
        t1y = (max.y - o.y) * invdir.y;
    } else {
        t0y = (max.y - o.y) * invdir.y;
        t1y = (min.y - o.y) * invdir.y;
    }
    if(invdir.z >= 0) {
        t0z = (min.z - o.z) * invdir.z;
        t1z = (max.z - o.z) * invdir.z;
    } else {
        t0z = (max.z - o.z) * invdir.z;
        t1z = (min.z - o.z) * invdir.z;
    }

    if((t0x > t1y) || (t0y > t1x)) return -1.0f;
    if(t0y > t0x) t0x = t0y;
    if(t1y < t1x) t1x = t1y;

    if((t0x > t1z) || (t0z > t1x)) return -1.0f;
    if(t0z > t0x) t0x = t0z;
    if(t1z < t1x) t1x = t1z;

    times[0] = t0x;
    times[1] = t1x;

    return 1.0f;
}

__host__ __device__ float triangleIntersectionTest(
    Geom geom,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& texCoord,
    glm::vec4* textures,
    bool& outside)
{
    const glm::vec3 p0 = glm::vec3(geom.transform * glm::vec4(geom.vertices[0], 1.f));
    const glm::vec3 p1 = glm::vec3(geom.transform * glm::vec4(geom.vertices[1], 1.f));
    const glm::vec3 p2 = glm::vec3(geom.transform * glm::vec4(geom.vertices[2], 1.f));
    const glm::vec3 n0 = glm::vec3(geom.transform * glm::vec4(geom.normals[0], 0.f));
    const glm::vec3 n1 = glm::vec3(geom.transform * glm::vec4(geom.normals[1], 0.f));
    const glm::vec3 n2 = glm::vec3(geom.transform * glm::vec4(geom.normals[2], 0.f));
    const glm::vec2 t0 = geom.uv[0];
    const glm::vec2 t1 = geom.uv[1];
    const glm::vec2 t2 = geom.uv[2];

    glm::vec3 edge1 = p1 - p0;
    glm::vec3 edge2 = p2 - p0;
    glm::vec3 pvec = glm::cross(r.direction, edge2);

    // If determinant is near zero, ray lies in plane of triangle
    float det = dot(edge1, pvec);

    if (det > -1e-8f && det < 1e-8f)
        return -1.0f;
    float inv_det = 1.0f / det;

    // Calculate distance from v[0] to ray origin
    glm::vec3 tvec = r.origin - p0;

    // Calculate U parameter and test bounds
    float u = dot(tvec, pvec) * inv_det;
    if (u < 0.0 || u > 1.0)
        return -1.0f;

    // Prepare to test V parameter
    glm::vec3 qvec = cross(tvec, edge1);

    // Calculate V parameter and test bounds
    float v = glm::dot(r.direction, qvec) * inv_det;
    if (v < 0.0 || u + v > 1.0)
        return -1.0f;

    // Ray intersects triangle -> compute t
    float t = dot(edge2, qvec) * inv_det;

    if (t < 0)
        return -1.0f;

    glm::vec3 bary(1.f - (u + v), u, v);

    // Compute the interpolated attributes
    intersectionPoint = r.origin + r.direction * t;
    normal = bary.x * n0 + bary.y * n1 + bary.z * n2;
    texCoord = bary.x * t0 + bary.y * t1 + bary.z * t2;

    if (geom.bumpmapTextureInfo.index > -1) {
        glm::vec3 b00 = glm::vec3(sampleBilinear(geom.bumpmapTextureInfo, texCoord, textures));
        glm::vec3 b01 = glm::vec3(sampleBilinear(geom.bumpmapTextureInfo, glm::vec2(texCoord.x + 1.f / geom.bumpmapTextureInfo.width, texCoord.y), textures));
        glm::vec3 b10 = glm::vec3(sampleBilinear(geom.bumpmapTextureInfo, glm::vec2(texCoord.x, texCoord.y + 1.f / geom.bumpmapTextureInfo.width), textures));

        normal = glm::normalize(normal + cross((b10 - b00), (b01 - b00)));
    }
    return t;
}

__host__ __device__ float boxIntersectionTest(
    Geom& box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    glm::vec2& texCoord,
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

        // Do the same as sphere uv coordinates for now (don't use cube mesh in general with image textures please)
        glm::vec3 uvP = getPointOnRay(q, tmin);
        texCoord = glm::vec2((atan2(uvP.y, uvP.x) + PI) / (2 * PI), (asin(uvP.z) + PI / 2) / PI);
        return glm::length(r.origin - intersectionPoint);
    }

    return -1.f;
}

__host__ __device__ float sphereIntersectionTest(
    Geom& sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    glm::vec2& texCoord,
    bool &outside)
{
    float radius = 0.5f;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2.f));
    if (radicand < 0.f)
    {
        return -1.f;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0.f;
    if (t1 < 0.f && t2 < 0.f)
    {
        return -1.f;
    }
    else if (t1 > 0.f && t2 > 0.f)
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
    
    glm::vec3 uvP = objspaceIntersection / radius;
    texCoord = glm::vec2((atan2(uvP.y, uvP.x) + PI) / (2.f * PI), (asin(uvP.z) + PI / 2.f) / PI);

    return glm::length(r.origin - intersectionPoint);
}
