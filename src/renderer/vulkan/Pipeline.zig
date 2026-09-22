const Self = @This();

const std = @import("std");
const vk = @import("api.zig").vk;
const Context = @import("Context.zig");
const shader_compile = @import("shader_compile.zig");

pub const Options = struct {
    context: *Context,
    vertex_fn: [:0]const u8,
    fragment_fn: [:0]const u8,
    step_fn: StepFunction = .per_vertex,
    topology: vk.PrimitiveTopology = .triangle_list,
    blending_enabled: bool = true,
    format: vk.Format,

    pub const StepFunction = enum { constant, per_vertex, per_instance };
};

context: *Context,
pipeline: vk.Pipeline,
stride: usize,

pub fn init(comptime VertexAttributes: ?type, opts: Options) !Self {
    const alloc = std.heap.c_allocator;
    const vertex = try shader_compile.module(opts.context, alloc, opts.vertex_fn, .vertex);
    defer opts.context.device.destroyShaderModule(vertex, null);
    const fragment = try shader_compile.module(opts.context, alloc, opts.fragment_fn, .fragment);
    defer opts.context.device.destroyShaderModule(fragment, null);

    const stages = [_]vk.PipelineShaderStageCreateInfo{
        shaderStage(.{ .vertex_bit = true }, vertex),
        shaderStage(.{ .fragment_bit = true }, fragment),
    };

    var attribute_storage: [16]vk.VertexInputAttributeDescription = undefined;
    const attribute_count: u32 = if (VertexAttributes) |T| count: {
        inline for (@typeInfo(T).@"struct".fields, 0..) |field, i| {
            attribute_storage[i] = .{
                .location = i,
                .binding = 0,
                .format = vertexFormat(field.type),
                .offset = @offsetOf(T, field.name),
            };
        }
        break :count @typeInfo(T).@"struct".fields.len;
    } else 0;

    var binding: vk.VertexInputBindingDescription = undefined;
    if (VertexAttributes) |T| {
        binding.binding = 0;
        binding.stride = @sizeOf(T);
        binding.input_rate = switch (opts.step_fn) {
            .per_instance, .constant => .instance,
            .per_vertex => .vertex,
        };
    }

    const vertex_input: vk.PipelineVertexInputStateCreateInfo = .{
        .vertex_binding_description_count = if (VertexAttributes == null) 0 else 1,
        .p_vertex_binding_descriptions = if (VertexAttributes == null) null else @ptrCast(&binding),
        .vertex_attribute_description_count = attribute_count,
        .p_vertex_attribute_descriptions = if (attribute_count == 0) null else &attribute_storage,
    };
    const input_assembly: vk.PipelineInputAssemblyStateCreateInfo = .{
        .topology = opts.topology,
        .primitive_restart_enable = .false,
    };
    const viewport_state: vk.PipelineViewportStateCreateInfo = .{
        .viewport_count = 1,
        .scissor_count = 1,
    };
    const rasterization: vk.PipelineRasterizationStateCreateInfo = .{
        .depth_clamp_enable = .false,
        .rasterizer_discard_enable = .false,
        .polygon_mode = .fill,
        .front_face = .counter_clockwise,
        .depth_bias_enable = .false,
        .depth_bias_constant_factor = 0,
        .depth_bias_clamp = 0,
        .depth_bias_slope_factor = 0,
        .line_width = 1,
    };
    const multisample: vk.PipelineMultisampleStateCreateInfo = .{
        .rasterization_samples = .{ .@"1_bit" = true },
        .sample_shading_enable = .false,
        .min_sample_shading = 0,
        .alpha_to_coverage_enable = .false,
        .alpha_to_one_enable = .false,
    };
    const blend_attachment: vk.PipelineColorBlendAttachmentState = .{
        .blend_enable = if (opts.blending_enabled) .true else .false,
        .src_color_blend_factor = .one,
        .dst_color_blend_factor = .one_minus_src_alpha,
        .color_blend_op = .add,
        .src_alpha_blend_factor = .one,
        .dst_alpha_blend_factor = .one_minus_src_alpha,
        .alpha_blend_op = .add,
        .color_write_mask = .{
            .r_bit = true,
            .g_bit = true,
            .b_bit = true,
            .a_bit = true,
        },
    };
    const color_blend: vk.PipelineColorBlendStateCreateInfo = .{
        .logic_op_enable = .false,
        .logic_op = .clear,
        .attachment_count = 1,
        .p_attachments = @ptrCast(&blend_attachment),
        .blend_constants = .{ 0, 0, 0, 0 },
    };
    const dynamic_states = [_]vk.DynamicState{ .viewport, .scissor };
    const dynamic: vk.PipelineDynamicStateCreateInfo = .{
        .dynamic_state_count = dynamic_states.len,
        .p_dynamic_states = &dynamic_states,
    };
    // Ghostty has one color attachment and uses Vulkan 1.3 dynamic rendering,
    // so pipelines do not need render-pass objects or framebuffers.
    const rendering: vk.PipelineRenderingCreateInfo = .{
        .view_mask = 0,
        .color_attachment_count = 1,
        .p_color_attachment_formats = @ptrCast(&opts.format),
        .depth_attachment_format = .undefined,
        .stencil_attachment_format = .undefined,
    };
    const info: vk.GraphicsPipelineCreateInfo = .{
        .p_next = &rendering,
        .stage_count = stages.len,
        .p_stages = &stages,
        .p_vertex_input_state = &vertex_input,
        .p_input_assembly_state = &input_assembly,
        .p_viewport_state = &viewport_state,
        .p_rasterization_state = &rasterization,
        .p_multisample_state = &multisample,
        .p_color_blend_state = &color_blend,
        .p_dynamic_state = &dynamic,
        .layout = opts.context.pipeline_layout,
        .subpass = 0,
        .base_pipeline_index = -1,
    };

    var pipeline: vk.Pipeline = undefined;
    _ = try opts.context.device.createGraphicsPipelines(
        .null_handle,
        @ptrCast(&info),
        null,
        @ptrCast(&pipeline),
    );
    return .{
        .context = opts.context,
        .pipeline = pipeline,
        .stride = if (VertexAttributes) |T| @sizeOf(T) else 0,
    };
}

pub fn deinit(self: Self) void {
    self.context.device.destroyPipeline(self.pipeline, null);
}

fn shaderStage(stage: vk.ShaderStageFlags, module_: vk.ShaderModule) vk.PipelineShaderStageCreateInfo {
    return .{ .stage = stage, .module = module_, .p_name = "main" };
}

fn vertexFormat(comptime T_: type) vk.Format {
    const T = switch (@typeInfo(T_)) {
        .@"struct" => |s| s.backing_integer.?,
        .@"enum" => |e| e.tag_type,
        else => T_,
    };
    const len, const Child = switch (@typeInfo(T)) {
        .array => |array| .{ array.len, array.child },
        else => .{ 1, T },
    };
    return switch (Child) {
        u8 => switch (len) {
            1 => .r8_uint,
            2 => .r8g8_uint,
            4 => .r8g8b8a8_uint,
            else => unreachable,
        },
        i8 => switch (len) {
            1 => .r8_sint,
            2 => .r8g8_sint,
            4 => .r8g8b8a8_sint,
            else => unreachable,
        },
        u16 => switch (len) {
            1 => .r16_uint,
            2 => .r16g16_uint,
            4 => .r16g16b16a16_uint,
            else => unreachable,
        },
        i16 => switch (len) {
            1 => .r16_sint,
            2 => .r16g16_sint,
            4 => .r16g16b16a16_sint,
            else => unreachable,
        },
        u32 => switch (len) {
            1 => .r32_uint,
            2 => .r32g32_uint,
            4 => .r32g32b32a32_uint,
            else => unreachable,
        },
        i32 => switch (len) {
            1 => .r32_sint,
            2 => .r32g32_sint,
            4 => .r32g32b32a32_sint,
            else => unreachable,
        },
        f32 => switch (len) {
            1 => .r32_sfloat,
            2 => .r32g32_sfloat,
            4 => .r32g32b32a32_sfloat,
            else => unreachable,
        },
        else => unreachable,
    };
}
