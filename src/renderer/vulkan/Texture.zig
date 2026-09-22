const Self = @This();

const vk = @import("api.zig").vk;
const Context = @import("Context.zig");
const bufferpkg = @import("buffer.zig");

pub const PixelFormat = enum { gray, rgba, bgra };

pub const Options = struct {
    context: *Context,
    format: vk.Format,
    upload_format: PixelFormat,
    min_filter: vk.Filter,
    mag_filter: vk.Filter,
    address_mode: vk.SamplerAddressMode,
    unnormalized_coordinates: bool = false,
};

context: *Context,
image: vk.Image,
memory: vk.DeviceMemory,
view: vk.ImageView,
sampler: vk.Sampler,
width: usize,
height: usize,
format: vk.Format,
upload_format: PixelFormat,

pub const Error = anyerror;

pub fn init(opts: Options, width: usize, height: usize, data: ?[]const u8) Error!Self {
    const image = try opts.context.device.createImage(&.{
        .image_type = .@"2d",
        .format = opts.format,
        .extent = .{ .width = @intCast(width), .height = @intCast(height), .depth = 1 },
        .mip_levels = 1,
        .array_layers = 1,
        .samples = .{ .@"1_bit" = true },
        .tiling = .optimal,
        .usage = .{
            .sampled_bit = true,
            .transfer_dst_bit = true,
            .transfer_src_bit = true,
            .color_attachment_bit = true,
        },
        .sharing_mode = .exclusive,
        .initial_layout = .undefined,
    }, null);
    errdefer opts.context.device.destroyImage(image, null);

    const requirements = opts.context.device.getImageMemoryRequirements(image);
    const memory = try opts.context.device.allocateMemory(&.{
        .allocation_size = requirements.size,
        .memory_type_index = try opts.context.memoryType(
            requirements.memory_type_bits,
            .{ .device_local_bit = true },
        ),
    }, null);
    errdefer opts.context.device.freeMemory(memory, null);
    try opts.context.device.bindImageMemory(image, memory, 0);

    const view = try opts.context.device.createImageView(&.{
        .image = image,
        .view_type = .@"2d",
        .format = opts.format,
        .components = .{
            .r = .identity,
            .g = .identity,
            .b = .identity,
            .a = .identity,
        },
        .subresource_range = colorRange(),
    }, null);
    errdefer opts.context.device.destroyImageView(view, null);

    const sampler = try opts.context.device.createSampler(&.{
        .mag_filter = opts.mag_filter,
        .min_filter = opts.min_filter,
        .mipmap_mode = .nearest,
        .address_mode_u = opts.address_mode,
        .address_mode_v = opts.address_mode,
        .address_mode_w = opts.address_mode,
        .mip_lod_bias = 0,
        .anisotropy_enable = .false,
        .max_anisotropy = 0,
        .compare_enable = .false,
        .compare_op = .never,
        .min_lod = 0,
        .max_lod = 0,
        .border_color = .float_transparent_black,
        .unnormalized_coordinates = if (opts.unnormalized_coordinates) .true else .false,
    }, null);
    errdefer opts.context.device.destroySampler(sampler, null);

    var self: Self = .{
        .context = opts.context,
        .image = image,
        .memory = memory,
        .view = view,
        .sampler = sampler,
        .width = width,
        .height = height,
        .format = opts.format,
        .upload_format = opts.upload_format,
    };
    errdefer self.deinit();
    try self.initialize(data);
    return self;
}

pub fn deinit(self: Self) void {
    self.context.device.destroySampler(self.sampler, null);
    self.context.device.destroyImageView(self.view, null);
    self.context.device.destroyImage(self.image, null);
    self.context.device.freeMemory(self.memory, null);
}

pub fn replaceRegion(self: Self, x: usize, y: usize, width: usize, height: usize, data: []const u8) Error!void {
    const staging = try bufferpkg.Handle.init(.{
        .context = self.context,
        .usage = .{ .transfer_src_bit = true },
    }, data.len);
    defer staging.deinit();
    try staging.write(0, data);

    const command_buffer = try self.context.beginCommands();
    const region: vk.BufferImageCopy = .{
        .buffer_offset = 0,
        .buffer_row_length = 0,
        .buffer_image_height = 0,
        .image_subresource = .{
            .aspect_mask = .{ .color_bit = true },
            .mip_level = 0,
            .base_array_layer = 0,
            .layer_count = 1,
        },
        .image_offset = .{ .x = @intCast(x), .y = @intCast(y), .z = 0 },
        .image_extent = .{ .width = @intCast(width), .height = @intCast(height), .depth = 1 },
    };
    self.context.device.cmdCopyBufferToImage(
        command_buffer,
        staging.buffer,
        self.image,
        .general,
        @ptrCast(&region),
    );
    imageBarrier(
        self.context,
        command_buffer,
        self.image,
        .{ .transfer_write_bit = true },
        .{ .shader_read_bit = true },
        .{ .transfer_bit = true },
        .{ .all_graphics_bit = true },
        .general,
        .general,
    );
    try self.context.submitCommands(command_buffer);
}

fn initialize(self: *Self, data: ?[]const u8) !void {
    const command_buffer = try self.context.beginCommands();
    imageBarrier(
        self.context,
        command_buffer,
        self.image,
        .{},
        if (data == null) .{ .shader_read_bit = true } else .{ .transfer_write_bit = true },
        .{ .top_of_pipe_bit = true },
        if (data == null) .{ .all_graphics_bit = true } else .{ .transfer_bit = true },
        // Textures remain in GENERAL for their lifetime. This trades some
        // layout-specific optimization for much simpler mixed render/sample/
        // transfer use across the generic renderer's passes.
        .undefined,
        .general,
    );
    try self.context.submitCommands(command_buffer);
    if (data) |bytes| try self.replaceRegion(0, 0, self.width, self.height, bytes);
}

pub fn colorRange() vk.ImageSubresourceRange {
    return .{
        .aspect_mask = .{ .color_bit = true },
        .base_mip_level = 0,
        .level_count = 1,
        .base_array_layer = 0,
        .layer_count = 1,
    };
}

pub fn imageBarrier(
    context: *Context,
    command_buffer: vk.CommandBuffer,
    image: vk.Image,
    src_access: vk.AccessFlags,
    dst_access: vk.AccessFlags,
    src_stage: vk.PipelineStageFlags,
    dst_stage: vk.PipelineStageFlags,
    old_layout: vk.ImageLayout,
    new_layout: vk.ImageLayout,
) void {
    const barrier: vk.ImageMemoryBarrier = .{
        .src_access_mask = src_access,
        .dst_access_mask = dst_access,
        .old_layout = old_layout,
        .new_layout = new_layout,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresource_range = colorRange(),
    };
    context.device.cmdPipelineBarrier(
        command_buffer,
        src_stage,
        dst_stage,
        .{},
        null,
        null,
        @ptrCast(&barrier),
    );
}
