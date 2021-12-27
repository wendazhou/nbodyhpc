#version 450

layout (location = 0) in float inDensity;
layout (location = 1) in vec3 inPosition;
layout (location = 2) in float inRadiusSquared;

layout (location = 0) out float outValue;

layout (origin_upper_left) in vec4 gl_FragCoord;

vec3 sample_locations_4[] = {
    vec3(-2, -6,  6) / 16,
    vec3( 6, -2,  2) / 16,
    vec3(-6,  2, -2) / 16,
    vec3( 2,  6, -6) / 16,
};

vec3 sample_locations_8[] = {
    vec3( 1, -3, -5) / 16,
    vec3(-1,  3,  5) / 16,
    vec3( 5,  1, -7) / 16,
    vec3(-3, -5,  3) / 16,
    vec3(-5,  5,  7) / 16,
    vec3(-7, -1, -3) / 16,
    vec3( 3,  7,  1) / 16,
    vec3( 7, -7, -1) / 16,
};

#define NUM_SAMPLES 8
#define SAMPLES_ARRAY_NAME_2(N) sample_locations_##N
#define SAMPLES_ARRAY_NAME(N) SAMPLES_ARRAY_NAME_2(N)
#define SAMPLES_ARRAY SAMPLES_ARRAY_NAME(NUM_SAMPLES)

void main() 
{
    const float increment = 1.0 / float(NUM_SAMPLES + 1);

    if (inRadiusSquared < 0.25) {
        // If the particle is too small, just return the density.
        outValue = inDensity;
        return;
    }

    vec3 delta = inPosition - vec3(gl_FragCoord.xy, 0);

    float overlap = 0.0;

    if (dot(delta, delta) < inRadiusSquared)
    {
        overlap += increment;
    }

    for(int i = 0; i < 4; i++)
    {
        vec3 sample_delta = delta + SAMPLES_ARRAY[i];
        float dist_squared = dot(sample_delta, sample_delta);

        if(dist_squared < inRadiusSquared)
        {
            overlap += increment;
        }
    }

    outValue = inDensity * overlap;
}