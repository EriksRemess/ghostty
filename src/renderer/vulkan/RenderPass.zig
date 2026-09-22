const Self = @This();

const vk = @import("api.zig").vk;
const Context = @import("Context.zig");
const bufferpkg = @import("buffer.zig");
const Pipeline = @import("Pipeline.zig");
const Sampler = @import("Sampler.zig");
const Target = @import("Target.zig");
const Texture = @import("Texture.zig");

pub const Options = struct {
    attachments: []const Attachment,

    pub const Attachment = struct {
        target: union(enum) { texture: Texture, target: Target },
        clear_color: ?[4]f32 = null,
    };
};

pub const Primitive = enum { triangle, triangle_strip };

pub const Step = struct {
    pipeline: Pipeline,
    uniforms: ?bufferpkg.Handle = null,
    buffers: []const ?bufferpkg.Handle = &.{},
    textures: []const ?Texture = &.{},
    samplers: []const ?Sampler = &.{},
    draw: Draw,

    pub const Draw = struct {
        type: Primitive,
        vertex_count: usize,
        instance_count: usize = 1,
    };
};

context: *Context,
command_buffer: vk.CommandBuffer,
descriptor_pool: vk.DescriptorPool,
attachment: Options.Attachment,
width: usize,
height: usize,

pub fn begin(
    context: *Context,
    command_buffer: vk.CommandBuffer,
    descriptor_pool: vk.DescriptorPool,
    opts: Options,
) Self {
    const attachment = opts.attachments[0];
    const texture: Texture = switch (attachment.target) {
        .texture => |value| value,
        .target => |value| value.texture,
    };

    Texture.imageBarrier(
        context,
        command_buffer,
        texture.image,
        .{ .shader_read_bit = true, .transfer_read_bit = true },
        .{ .color_attachment_write_bit = true },
        .{ .all_graphics_bit = true, .transfer_bit = true },
        .{ .color_attachment_output_bit = true },
        .general,
        .general,
    );

    var color_attachment: vk.RenderingAttachmentInfo = .{
        .image_view = texture.view,
        .image_layout = .general,
        .resolve_mode = .{},
        .resolve_image_layout = .undefined,
        .load_op = if (attachment.clear_color != null) .clear else .load,
        .store_op = .store,
        .clear_value = .{ .color = .{ .float_32 = .{ 0, 0, 0, 0 } } },
    };
    if (attachment.clear_color) |color| {
        color_attachment.clear_value.color.float_32 = color;
    }

    const rendering: vk.RenderingInfo = .{
        .render_area = .{
            .offset = .{ .x = 0, .y = 0 },
            .extent = .{ .width = @intCast(texture.width), .height = @intCast(texture.height) },
        },
        .layer_count = 1,
        .view_mask = 0,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_attachment),
    };
    context.device.cmdBeginRendering(command_buffer, &rendering);

    // A negative viewport height converts Vulkan's framebuffer coordinates to
    // the +Y-down convention used by Ghostty's existing renderer interface.
    const viewport: vk.Viewport = .{
        .x = 0,
        .y = @floatFromInt(texture.height),
        .width = @floatFromInt(texture.width),
        .height = -@as(f32, @floatFromInt(texture.height)),
        .min_depth = 0,
        .max_depth = 1,
    };
    context.device.cmdSetViewport(command_buffer, 0, @ptrCast(&viewport));

    const scissor: vk.Rect2D = .{
        .offset = .{ .x = 0, .y = 0 },
        .extent = .{ .width = @intCast(texture.width), .height = @intCast(texture.height) },
    };
    context.device.cmdSetScissor(command_buffer, 0, @ptrCast(&scissor));

    return .{
        .context = context,
        .command_buffer = command_buffer,
        .descriptor_pool = descriptor_pool,
        .attachment = attachment,
        .width = texture.width,
        .height = texture.height,
    };
}

pub fn step(self: *Self, step_: Step) void {
    if (step_.draw.instance_count == 0) return;
    _ = step_.draw.type;

    const alloc_info: vk.DescriptorSetAllocateInfo = .{
        .descriptor_pool = self.descriptor_pool,
        .descriptor_set_count = 1,
        .p_set_layouts = @ptrCast(&self.context.descriptor_set_layout),
    };
    var descriptor_set: vk.DescriptorSet = undefined;
    self.context.device.allocateDescriptorSets(&alloc_info, @ptrCast(&descriptor_set)) catch return;

    var writes: [4]vk.WriteDescriptorSet = undefined;
    var buffer_infos: [2]vk.DescriptorBufferInfo = undefined;
    var image_infos: [2]vk.DescriptorImageInfo = undefined;
    var write_count: usize = 0;
    var buffer_count: usize = 0;
    var image_count: usize = 0;

    if (step_.uniforms) |buffer| {
        buffer_infos[buffer_count] = .{ .buffer = buffer.buffer, .offset = 0, .range = vk.WHOLE_SIZE };
        writes[write_count] = descriptorWrite(descriptor_set, 0, .uniform_buffer);
        writes[write_count].p_buffer_info = @ptrCast(&buffer_infos[buffer_count]);
        write_count += 1;
        buffer_count += 1;
    }
    if (step_.buffers.len > 1) if (step_.buffers[1]) |buffer| {
        buffer_infos[buffer_count] = .{ .buffer = buffer.buffer, .offset = 0, .range = vk.WHOLE_SIZE };
        writes[write_count] = descriptorWrite(descriptor_set, 1, .storage_buffer);
        writes[write_count].p_buffer_info = @ptrCast(&buffer_infos[buffer_count]);
        write_count += 1;
        buffer_count += 1;
    };
    for (step_.textures, 0..) |texture_, i| if (texture_) |texture| {
        if (i >= image_infos.len) break;
        const sampler = if (i < step_.samplers.len and step_.samplers[i] != null)
            step_.samplers[i].?.sampler
        else
            texture.sampler;
        image_infos[image_count] = .{
            .sampler = sampler,
            .image_view = texture.view,
            .image_layout = .general,
        };
        writes[write_count] = descriptorWrite(descriptor_set, @intCast(2 + i), .combined_image_sampler);
        writes[write_count].p_image_info = @ptrCast(&image_infos[image_count]);
        write_count += 1;
        image_count += 1;
    };
    if (write_count > 0) self.context.device.updateDescriptorSets(writes[0..write_count], null);

    self.context.device.cmdBindPipeline(self.command_buffer, .graphics, step_.pipeline.pipeline);
    self.context.device.cmdBindDescriptorSets(
        self.command_buffer,
        .graphics,
        self.context.pipeline_layout,
        0,
        @ptrCast(&descriptor_set),
        null,
    );
    if (step_.buffers.len > 0) if (step_.buffers[0]) |vertex| {
        const offset: vk.DeviceSize = 0;
        self.context.device.cmdBindVertexBuffers(
            self.command_buffer,
            0,
            @ptrCast(&vertex.buffer),
            @ptrCast(&offset),
        );
    };
    self.context.device.cmdDraw(
        self.command_buffer,
        @intCast(step_.draw.vertex_count),
        @intCast(step_.draw.instance_count),
        0,
        0,
    );
}

pub fn complete(self: *const Self) void {
    self.context.device.cmdEndRendering(self.command_buffer);
    const texture: Texture = switch (self.attachment.target) {
        .texture => |value| value,
        .target => |value| value.texture,
    };
    Texture.imageBarrier(
        self.context,
        self.command_buffer,
        texture.image,
        .{ .color_attachment_write_bit = true },
        .{ .shader_read_bit = true, .transfer_read_bit = true },
        .{ .color_attachment_output_bit = true },
        .{ .all_graphics_bit = true, .transfer_bit = true },
        .general,
        .general,
    );
}

fn descriptorWrite(set: vk.DescriptorSet, binding: u32, descriptor_type: vk.DescriptorType) vk.WriteDescriptorSet {
    return .{
        .dst_set = set,
        .dst_binding = binding,
        .dst_array_element = 0,
        .descriptor_count = 1,
        .descriptor_type = descriptor_type,
        .p_image_info = undefined,
        .p_buffer_info = undefined,
        .p_texel_buffer_view = undefined,
    };
}
