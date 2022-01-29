// This source file contains inline bytecode for the shaders.

#include "shaders.h"

namespace {
#include "shaders/triangle.frag.spv.h"
#include "shaders/triangle.vert.spv.h"
} // namespace

namespace wenda {
namespace vulkan {
tcb::span<const unsigned char> get_vertex_shader_bytecode() {
    return tcb::span<const unsigned char>(triangle_vert_spv);
}

tcb::span<const unsigned char> get_fragment_shader_bytecode() {
    return tcb::span<const unsigned char>(triangle_frag_spv);
}
} // namespace vulkan
} // namespace wenda
