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



// From https://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/raytri/ (Algorithm 2)
__host__ __device__ bool intersectTriangle(
    const glm::vec3& orig,
    const glm::vec3& dir,
    const glm::vec3& v0,
    const glm::vec3& v1,
    const glm::vec3& v2,
    float& t,
	glm::vec2& triPos)
{
    //double edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
    //double det, inv_det;

    const float eps = 0.000001f;

    /* find vectors for two edges sharing vert0 */
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;

    /* begin calculating determinant - also used to calculate U parameter */
	glm::vec3 pvec = glm::cross(dir, edge2);

    /* if determinant is near zero, ray lies in plane of triangle */
	float det = glm::dot(edge1, pvec);

    /* calculate distance from vert0 to ray origin */
	glm::vec3 tvec = orig - v0;
    float inv_det = 1.0f / det;

    glm::vec3 qvec;
    if (det > eps)
    {
        /* calculate U parameter and test bounds */
		triPos.x = glm::dot(tvec, pvec);
        if (triPos.x < 0.0 || triPos.x > det)
            return false;

        /* prepare to test V parameter */
		qvec = glm::cross(tvec, edge1);

        /* calculate V parameter and test bounds */
		triPos.y = glm::dot(dir, qvec);
		if (triPos.y < 0.0 || triPos.x + triPos.y > det)
			return false;


    }
    else if (det < -eps)
    {
        /* calculate U parameter and test bounds */
		triPos.x = glm::dot(tvec, pvec);
		if (triPos.x > 0.0 || triPos.x < det)
			return false;

        /* prepare to test V parameter */
		qvec = glm::cross(tvec, edge1);

        /* calculate V parameter and test bounds */
		triPos.y = glm::dot(dir, qvec);
		if (triPos.y > 0.0 || triPos.x + triPos.y < det)
			return false;
    }
    else return false;  /* ray is parallell to the plane of the triangle */

    /* calculate t, ray intersects triangle */
	t = glm::dot(edge2, qvec) * inv_det;
	triPos *= inv_det;
    return true;
}

__host__ __device__ float meshIntersectionTest(
    Geom mesh,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    Triangle* triangles) {

    Ray rt;
    rt.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    rt.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float t;
    float tmin = FLT_MAX;
    glm::vec2 triPos;
	glm::vec2 minTriPos;
    int minIdx;

    int maxIndex = mesh.triangleStartIdx + mesh.triangleCount;
    for (int i = mesh.triangleStartIdx; i < maxIndex; ++i)
    {
        const glm::vec3 v1 = triangles[i].v1;
        const glm::vec3 v2 = triangles[i].v2;
        const glm::vec3 v3 = triangles[i].v3;
        if (!intersectTriangle(
            rt.origin,
            rt.direction,
            v1,
            v2,
            v3,
            t, triPos))
        {
            continue;
        }

        if (t < tmin)
        {
            tmin = t;
            minTriPos = triPos;
            minIdx = i;
        }
    }

    if (tmin == FLT_MAX) return -1;

    glm::vec3 intersectionPointLocal = getPointOnRay(rt, tmin);
    glm::vec3 normalLocal = glm::normalize(glm::cross(triangles[minIdx].v2 - triangles[minIdx].v1, triangles[minIdx].v3 - triangles[minIdx].v1));

    intersectionPoint = multiplyMV(mesh.transform, glm::vec4(intersectionPointLocal, 1.f));
    normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(normalLocal, 0.f)));

    outside = glm::dot(normal, r.direction) < 0;

    return t;
}