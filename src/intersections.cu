#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    const Geom& box,                // Represents the geometry of the box, its transformation matrices (inverseTransform, transform, etc.)
    const Ray& r,                   // The ray in world space that may intersect with the box
    glm::vec3 &intersectionPoint,   // A reference that will store the world-space intersection point if the ray intersects the box
    glm::vec3 &normal,              // A reference that will store the world-space surface normal at the intersection point
    bool &outside)                  // A reference to a boolean that will indicate if the intersection occurs from outside the box
{
    // 1: Transform Ray from World Space into Object Space
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    // 2: Initialize Variables for Ray-Box Intersection
    float tmin = -1e38f;    // Largest value of t where the ray enters the box, after considering all three axes (x, y, z)
    float tmax = 1e38f;     // Smallest value of t where the ray exits the box, after considering all three axes (x, y, z)
    glm::vec3 tmin_n;       // Normal of the box at tmin
    glm::vec3 tmax_n;       // Normal of the box at tmax

    // 3: Iterate Over Each Axis (x, y, z)
    // It is worth noting that Unscaled, the cube ranges from -0.5 to 0.5 in each axis
    //
    // For each axis, the function calculates the possible intersection points by determining where the ray intersects the box's planes (two planes for each axis)
    // Since the box is axis-aligned, it has faces perpendicular to each axis
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];                 // The ray’s direction component along axis xyz in object space
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz; // The distance along the ray where it intersects the first plane (e.g. at x = -0.5)
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz; // The distance along the ray where it intersects the second plane (e.g. at x = +0.5)
            float ta = min(t1, t2);                     // The smaller of the two distances
            float tb = max(t1, t2);                     // The larger of the two distances
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;                 // The normal of the surface at the intersection (depending on whether the ray hits from the positive or negative side)

            // 4: Update tmin and tmax based on the current axis' intersections
            if (ta > RAY_TRACE_EPSILON && ta > tmin)
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

    // 5: Check for Valid Intersection
    if (tmax >= tmin && tmax > RAY_TRACE_EPSILON)
    {
        outside = true;
        if (tmin <= RAY_TRACE_EPSILON)
        {
            // If tmin is negative but tmax is positive, the ray starts inside the box and exits at tmax, so the intersection happens at tmax
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }

        // 6: Calculate Intersection Point and Normal in World Space
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));

        // 7: Return the distance between the ray’s origin and the intersection point in world space
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    const Geom& sphere,             // Represents the geometry of the sphere, its transformation matrices (inverseTransform, transform, etc.)
    const Ray& r,                   // The ray in world space that may intersect with the sphere
    glm::vec3 &intersectionPoint,   // A reference that will store world-space the intersection point if the ray intersects the sphere
    glm::vec3 &normal,              // A reference that will store world-space the surface normal at the intersection point
    bool &outside)                  // A reference to a boolean that will indicate if the intersection occurs from outside the sphere
{
    // 1: Set Sphere's Unscaled Radius
    // Unscaled, the sphere has radius 0.5
    float radius = .5;

    // 2: Transform Ray from World Space into Object Space
    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;  // normalized

    // 3: Solve Quadratic for Intersection
    // The quadratic formula is used to find the intersection points between the ray and the sphere
    // This is derived from the equation of the sphere and the parametric equation of the ray
    //
    // Equation of Sphere with radius r centered at the origin: x^2 + y^2 + z^2 = r^2, or ||p||^2 = r^2 in vector form
    // Equation of parameterized Ray: p(t) = ro + t * rd
    // Substituting Ray into Sphere Equation: r^2 = ||p(t)||^2 = ||ro + t * rd||^2 = (ro + t * rd) DOT (ro + t * rd) = ro DOT ro + 2t(ro DOT rd) + t^2(rd DOT rd)
    // Since rd is normalized, rd DOT rd = 1, so r^2 = ro DOT ro + 2t(ro DOT rd) + t^2
    // Which gives our quadratic equation in t is: t^2 + 2(ro DOT rd)t + (ro DOT ro - r^2) = 0
    // Which gives us a = 1, b = 2(ro DOT rd), and c = ro DOT ro - r^2 for the standard form of a quadratic equation at^2 + bt + c = 0
    //
    // To solve for t, we use the equation t = (-b ± sqrt(b^2 - 4ac)) / 2a
    // Substituting the variables gives: t = (-2(ro DOT rd) ± sqrt((2(ro DOT rd))^2 - 4(ro DOT ro - r^2))) / 2
    // Simplifying to: t = -(ro DOT rd) ± sqrt((ro DOT rd)^2 - (ro DOT ro - r^2))

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    // 4: Calculate Parametric Intersection Values
    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < RAY_TRACE_EPSILON && t2 < RAY_TRACE_EPSILON)
    {
        // both intersections behind ray
        return -1;
    }
    else if (t1 > RAY_TRACE_EPSILON && t2 > RAY_TRACE_EPSILON)
    {
        // both intersections in front of ray
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        // one intersection behind ray and one intersection in front of ray
        t = max(t1, t2);
        outside = false;
    }

    // 5: Calculate Intersection Point and Normal in World Space

    // Calculate the Intersection Point in Object Space
    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    // Transform intersection point and normal from the object space of the sphere into world space
    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));

    // 6: Adjust Normal for Inside Intersection
    if (!outside)
    {
        normal = -normal;
    }

    // 7: Return the distance between the ray’s origin and the intersection point in world space
    return glm::length(r.origin - intersectionPoint);
}

// Derived from glm::intersectRayTriangle
// Using Möller–Trumbore intersection algorithm
__host__ __device__ float triangleIntersectionTest(
    const Geom& triangle,
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    glm::vec3 baryPosition = glm::vec3(0.0f);

    glm::vec3 e1 = triangle.v1 - triangle.v0;
    glm::vec3 e2 = triangle.v2 - triangle.v0;

    glm::vec3 p = glm::cross(r.direction, e2);

    float a = glm::dot(e1, p);

    if (-FLT_EPSILON < a && a < FLT_EPSILON)
        return -1;

    float f = 1.0f / a;

    glm::vec3 s = r.origin - triangle.v0;
    baryPosition.x = f * glm::dot(s, p);
    if (baryPosition.x < 0.0f || baryPosition.x > 1.0f)
        return -1;

    glm::vec3 q = glm::cross(s, e1);
    baryPosition.y = f * glm::dot(r.direction, q);
    if (baryPosition.y < 0.0f || baryPosition.y + baryPosition.x > 1.0f)
        return -1;

    baryPosition.z = f * glm::dot(e2, q);

	if (baryPosition.z < 0.0f)
		return -1;

    // At this point, we have an intersection
	float t = baryPosition.z;

	glm::vec3 interpolatedNormal = glm::normalize(
		baryPosition.x * triangle.n0 +
		baryPosition.y * triangle.n1 +
		baryPosition.z * triangle.n2);

	outside = glm::dot(-r.direction, interpolatedNormal) >= 0;
	intersectionPoint = getPointOnRay(r, t);
	normal = outside ? interpolatedNormal : -interpolatedNormal;

	return t;
}

__host__ __device__ bool aabbIntersectionTest(const AABB& box, const Ray& ray)
{
    glm::vec3 invDir = 1.0f / ray.direction;

    // Calculate intersection with the slabs in the x direction
    float tx0 = (box.min.x - ray.origin.x) * invDir.x;
    float tx1 = (box.max.x - ray.origin.x) * invDir.x;
    float tmin = min(tx0, tx1);
    float tmax = max(tx0, tx1);

    // Early exit: check if the x-axis slabs already disqualify an intersection
    if (tmax < tmin) return false;

    // Calculate intersection with the slabs in the y direction
    float ty0 = (box.min.y - ray.origin.y) * invDir.y;
    float ty1 = (box.max.y - ray.origin.y) * invDir.y;
    tmin = max(tmin, min(ty0, ty1));
    tmax = min(tmax, max(ty0, ty1));

    // Early exit: if no overlap in the y direction, return false
    if (tmax < tmin) return false;

    // Calculate intersection with the slabs in the z direction
    float tz0 = (box.min.z - ray.origin.z) * invDir.z;
    float tz1 = (box.max.z - ray.origin.z) * invDir.z;
    tmin = max(tmin, min(tz0, tz1));
    tmax = min(tmax, max(tz0, tz1));

    // Final check: return true if there is an intersection, otherwise false
    return tmax >= tmin;
}
