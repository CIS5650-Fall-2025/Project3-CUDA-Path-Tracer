__host__ __device__ bool triangleIntersectionTest(
    Triangle triangle,
    Ray r,
    glm::vec2 triPos,
    float& t)
{
    Ray rt;
    rt.origin = multiplyMV(triangle.inverseTransform, glm::vec4(r.origin, 1.0f));
    rt.direction = glm::normalize(multiplyMV(triangle.inverseTransform, glm::vec4(r.direction, 0.0f)));

    if (glm::intersectRayTriangle(rt.origin, rt.direction, triangle.v1, triangle.v2, triangle.v2, triPos, t))
        return true;
    else return false;
}

__host__ __device__ float meshIntersectionTest(
    geom mesh,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    Ray rt;
    rt.origin = multiplyMV(triangle.inverseTransform, glm::vec4(r.origin, 1.0f));
    rt.direction = glm::normalize(multiplyMV(triangle.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float t;
    glm::vec2 triPos;

    int maxIndex = mesh.triangletartIdx + mesh.triangleCount;
    for (int i = mesh.triangleStartIdx; i < maxIndex; ++i)
    {
        if (!triangleIntersectionTest(triangleBuffer[i], rt, intersectionPoint, normal, outside, triPos, t))
            continue;

        glm::vec3 intersectionPointLocal = getPointOnRay(rt, tmin);
        glm::vec3 normalLocal = glm::normalize(glm::cross(triangle.v2 - triangle.v1, triangle.v3 - triangle.v2));

        intersectionPoint = multiplyMV(sphere.transform, glm::vec4(intersectionPointLocal, 1.f));
        normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(normalLocal, 0.f)));

        outside = glm::dot(normal, r.direction) < 0;

        return t;
    }
    return -1;
}