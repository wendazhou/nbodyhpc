#pragma once

#include <span.hpp>

namespace wenda {

namespace vulkan {
tcb::span<const unsigned char> get_vertex_shader_bytecode();
tcb::span<const unsigned char> get_fragment_shader_bytecode();
} // namespace vulkan

} // namespace wenda