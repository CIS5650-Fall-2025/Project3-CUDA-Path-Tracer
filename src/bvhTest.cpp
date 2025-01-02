

for (int i = 0; i < len(scene.geoms), ++i)
{
    switch scene.geoms[i].type

    case SPHERE:

    case CUBE:

    case MESH:
}



struct BVHNode
{
    glm::vec3 minPoint, maxPoint;
    unsigned int leftFirst, numTriangles;
};

glm::vec3 triangleCenter;
BVHNode bvhData[2 * numTriangles - 1];

unsigned int rootNodeIdx = 0;
unsigned int nodesUsed = 1;

 
void BuildBVH()
{
    for (int i = 0; i < N; i++) tri[i].centroid = 
        (tri[i].vertex0 + tri[i].vertex1 + tri[i].vertex2) * 0.3333f;
    // assign all triangles to root node
    BVHNode& root = bvhNode[rootNodeIdx];
    root.leftChild = root.rightChild = 0;
    root.firstPrim = 0, root.primCount = N;
    updateNodeBounds( rootNodeIdx );
    // subdivide recursively
    subdivide( rootNodeIdx );
}

void intersectBVH( Ray& ray, const unsigned int nodeIdx )
{
	BVHNode& node = bvhNode[nodeIdx];
	if (!intersectAABB( ray, node.aabbMin, node.aabbMax )) return;
	if (node.isLeaf())
	{
		for (unsigned int i = 0; i < node.triCount; i++ )
			intersectTri( ray, tri[triIdx[node.leftFirst + i]] );
	}
	else
	{
		intersectBVH( ray, node.leftFirst );
		intersectBVH( ray, node.leftFirst + 1 );
	}
}

void buildBVH()
{
	// populate triangle index array
	for (int i = 0; i < N; i++) triIdx[i] = i;
	// calculate triangle centroids for partitioning
	for (int i = 0; i < N; i++)
		tri[i].centroid = (tri[i].vertex0 + tri[i].vertex1 + tri[i].vertex2) * 0.3333f;
	// assign all triangles to root node
	BVHNode& root = bvhNode[rootNodeIdx];
	root.leftFirst = 0, root.triCount = N;
	updateNodeBounds( rootNodeIdx );
	// subdivide recursively
	subdivide( rootNodeIdx );
}

void subdivide( unsigned int nodeIdx )
{
	// terminate recursion
	BVHNode& node = bvhNode[nodeIdx];
	if (node.triCount <= 2) return;
	// determine split axis and position
	float3 extent = node.aabbMax - node.aabbMin;
	int axis = 0;
	if (extent.y > extent.x) axis = 1;
	if (extent.z > extent[axis]) axis = 2;
	float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;
	// in-place partition
	int i = node.leftFirst;
	int j = i + node.triCount - 1;
	while (i <= j)
	{
		if (tri[triIdx[i]].centroid[axis] < splitPos)
			i++;
		else
			swap( triIdx[i], triIdx[j--] );
	}
	// abort split if one of the sides is empty
	int leftCount = i - node.leftFirst;
	if (leftCount == 0 || leftCount == node.triCount) return;
	// create child nodes
	int leftChildIdx = nodesUsed++;
	int rightChildIdx = nodesUsed++;
	bvhNode[leftChildIdx].leftFirst = node.leftFirst;
	bvhNode[leftChildIdx].triCount = leftCount;
	bvhNode[rightChildIdx].leftFirst = i;
	bvhNode[rightChildIdx].triCount = node.triCount - leftCount;
	node.leftFirst = leftChildIdx;
	node.triCount = 0;
	updateNodeBounds( leftChildIdx );
	updateNodeBounds( rightChildIdx );
	// recurse
	subdivide( leftChildIdx );
	subdivide( rightChildIdx );
}