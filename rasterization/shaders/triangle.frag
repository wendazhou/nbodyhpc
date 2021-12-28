#version 450

layout (location = 0) in float inDensity;
layout (location = 1) in vec3 inPosition;
layout (location = 2) in float inRadiusSquared;

layout (location = 0) out float outValue;

layout (origin_upper_left) in vec4 gl_FragCoord;

layout (constant_id=0) const int SUBGRID_SIZE = 4;


void main() 
{
    const float increment = 1.0 / float(SUBGRID_SIZE * SUBGRID_SIZE * SUBGRID_SIZE);

    if (inRadiusSquared < 0.25) {
        // If the particle is too small, just return the density.
        outValue = inDensity;
        return;
    }

    // adjust delta so it computes offset from corner of cell
    vec3 delta = inPosition - vec3(gl_FragCoord.xy - 0.5, -0.5);
    float overlap = 0.0;

    for(int i = 0; i < SUBGRID_SIZE; i++) {
        float x_offset = (float(i) + 0.5) / SUBGRID_SIZE;

        for(int j = 0; j < SUBGRID_SIZE; ++j) {
            float y_offset = (float(j) + 0.5) / SUBGRID_SIZE;

            for(int k = 0; k < SUBGRID_SIZE; ++k) {
                float z_offset = (float(k) + 0.5) / SUBGRID_SIZE;
                vec3 subdelta = delta - vec3(x_offset, y_offset, z_offset);

                if (dot(subdelta, subdelta) < inRadiusSquared) {
                    overlap += increment;
                }
            }
        }
    }

    outValue = inDensity * overlap;
}