#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in float inWeight;
layout (location = 2) in float inRadius;

layout (location = 0) out float outDensity;
layout (location = 1) out vec3 outPosition;
layout (location = 2) out float outRadiusSquared;

out gl_PerVertex {
	vec4 gl_Position;
    float gl_PointSize;
	float gl_ClipDistance[];
};

layout(push_constant) uniform PushConsts {
	float boxSize;		// size of the box (in arbitrary units)
    float viewportSize; // size of the viewport (in pixels)
	float planeDepth;   // depth of the current plane (in arbitrary units)
} pushConsts;

void main() 
{
	float line_element = pushConsts.viewportSize / pushConsts.boxSize;

	// compute effective radius of circle at given height
	// this is also used to check whether the point is inside the plane being rendered.
	float z_offset = inPos.z - pushConsts.planeDepth;

	gl_ClipDistance[0] = line_element * (inRadius - abs(z_offset)) + 1;

	if (gl_ClipDistance[0] < 0)
	{
		return;
	}

	float out_radius = inRadius * line_element;

	float plane_radius = sqrt(max(0.0, inRadius * inRadius - z_offset * z_offset));
	float n_pixel_diameter = 2 * ceil(plane_radius * line_element);
	float volume = 4. / 3. * radians(180) * out_radius * out_radius * out_radius;

	if (out_radius < 0.5) {
		outDensity = inWeight;
		gl_PointSize = 1;
	}
	else {
		outDensity = inWeight / volume;
    	gl_PointSize = n_pixel_diameter + 2;
	}

	gl_Position = vec4(2 * (inPos.xy / pushConsts.boxSize - 0.5), 0.0, 1.0);
	outRadiusSquared = out_radius * out_radius;
	outPosition = vec3(inPos.xy, inPos.z - pushConsts.planeDepth) * line_element;
}
