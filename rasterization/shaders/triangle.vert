#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in float inWeight;
layout (location = 2) in float inRadius;

layout (location = 0) out float outDensity;

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
	float line_element = pushConsts.boxSize / pushConsts.viewportSize;
	float volume_element = line_element * line_element * line_element;

	// compute effective radius of circle at given height
	// this is also used to check whether the point is inside the plane being rendered.
	float z_offset = inPos.z - pushConsts.planeDepth;
	float out_radius_squared = inRadius * inRadius - z_offset * z_offset;
	gl_ClipDistance[0] = out_radius_squared;
	float out_radius = sqrt(max(out_radius_squared, 1e-12));

	outDensity = inWeight / (4 / 3 * radians(180) * inRadius * inRadius * inRadius) * volume_element;
	gl_Position = vec4(2 * (inPos.xy / pushConsts.boxSize - 0.5), 0.0, 1.0);
    gl_PointSize = 2 * out_radius / line_element;
}
