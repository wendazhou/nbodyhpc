#version 450

layout (location = 0) in float inDensity;
layout (location = 0) out float outValue;


void main() 
{
    bool isInside = distance(gl_PointCoord, vec2(0.5, 0.5)) < 0.5;
    if (isInside) {
        outValue = inDensity;
    }
    else {
        discard;
    }
}