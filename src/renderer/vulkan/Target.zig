//! Represents an offscreen Vulkan render target.
//!
//! texture is the color attachment. Every completed frame is copied to the
//! host-visible readback buffer so CPU presentation is always available. If
//! external-memory support is usable, the same frame is also copied to
//! export_image, whose memory is exported to GTK as a dma-buf.

const Self = @This();

const std = @import("std");
const vk = @import("api.zig").vk;
const Context = @import("Context.zig");
const Texture = @import("Texture.zig");
const bufferpkg = @import("buffer.zig");
const Dmabuf = @import("../Dmabuf.zig");

const log = std.log.scoped(.vulkan);
const drm_format_abgr8888: u32 = @as(u32, 'A') |
    (@as(u32, 'B') << 8) |
    (@as(u32, '2') << 16) |
    (@as(u32, '4') << 24);
const drm_format_mod_linear: u64 = 0;

pub const Options = struct {
    context: *Context,
    width: usize,
    height: usize,
    format: vk.Format,
};

context: *Context,
texture: Texture,
readback: bufferpkg.Handle,
export_image: ?ExportImage,
width: usize,
height: usize,

pub fn init(opts: Options) !Self {
    const texture = try Texture.init(.{
        .context = opts.context,
        .format = opts.format,
        .upload_format = .rgba,
        .min_filter = .linear,
        .mag_filter = .linear,
        .address_mode = .clamp_to_edge,
    }, opts.width, opts.height, null);
    errdefer texture.deinit();

    const readback = try bufferpkg.Handle.init(.{
        .context = opts.context,
        .usage = .{ .transfer_dst_bit = true },
    }, opts.width * opts.height * 4);
    errdefer readback.deinit();

    const export_image: ?ExportImage = if (opts.context.supports_dmabuf)
        ExportImage.init(opts.context, opts.width, opts.height, opts.format) catch |err| fallback: {
            // Modifier support can differ by format even when all required
            // extensions exist. Retain the guaranteed CPU fallback.
            log.warn("DMA-BUF export image unavailable, using memory presentation err={}", .{err});
            break :fallback null;
        }
    else
        null;
    errdefer if (export_image) |value| value.deinit();

    return .{
        .context = opts.context,
        .texture = texture,
        .readback = readback,
        .export_image = export_image,
        .width = opts.width,
        .height = opts.height,
    };
}

pub fn deinit(self: *Self) void {
    if (self.export_image) |value| value.deinit();
    self.readback.deinit();
    self.texture.deinit();
}

pub fn recordReadback(self: *const Self, command_buffer: vk.CommandBuffer) void {
    // Both presentation paths consume the rendered attachment, so establish
    // transfer visibility before recording either copy.
    Texture.imageBarrier(
        self.context,
        command_buffer,
        self.texture.image,
        .{ .color_attachment_write_bit = true },
        .{ .transfer_read_bit = true },
        .{ .color_attachment_output_bit = true },
        .{ .transfer_bit = true },
        .general,
        .general,
    );

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
        .image_offset = .{ .x = 0, .y = 0, .z = 0 },
        .image_extent = .{
            .width = @intCast(self.width),
            .height = @intCast(self.height),
            .depth = 1,
        },
    };
    self.context.device.cmdCopyImageToBuffer(
        command_buffer,
        self.texture.image,
        .general,
        self.readback.buffer,
        @ptrCast(&region),
    );

    if (self.export_image) |export_image| {
        // The export image is deliberately separate: dma-buf-capable tiling
        // and memory requirements need not match the optimal render target.
        Texture.imageBarrier(
            self.context,
            command_buffer,
            export_image.image,
            .{ .memory_read_bit = true },
            .{ .transfer_write_bit = true },
            .{ .all_commands_bit = true },
            .{ .transfer_bit = true },
            .general,
            .general,
        );
        const image_copy: vk.ImageCopy = .{
            .src_subresource = .{
                .aspect_mask = .{ .color_bit = true },
                .mip_level = 0,
                .base_array_layer = 0,
                .layer_count = 1,
            },
            .src_offset = .{ .x = 0, .y = 0, .z = 0 },
            .dst_subresource = .{
                .aspect_mask = .{ .color_bit = true },
                .mip_level = 0,
                .base_array_layer = 0,
                .layer_count = 1,
            },
            .dst_offset = .{ .x = 0, .y = 0, .z = 0 },
            .extent = .{
                .width = @intCast(self.width),
                .height = @intCast(self.height),
                .depth = 1,
            },
        };
        self.context.device.cmdCopyImage(
            command_buffer,
            self.texture.image,
            .general,
            export_image.image,
            .general,
            @ptrCast(&image_copy),
        );
        Texture.imageBarrier(
            self.context,
            command_buffer,
            export_image.image,
            .{ .transfer_write_bit = true },
            .{ .memory_read_bit = true },
            .{ .transfer_bit = true },
            .{ .bottom_of_pipe_bit = true },
            .general,
            .general,
        );
    }

    // Make the transfer visible to mapMemory after the submission fence.
    const barrier: vk.BufferMemoryBarrier = .{
        .src_access_mask = .{ .transfer_write_bit = true },
        .dst_access_mask = .{ .host_read_bit = true },
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .buffer = self.readback.buffer,
        .offset = 0,
        .size = vk.WHOLE_SIZE,
    };
    self.context.device.cmdPipelineBarrier(
        command_buffer,
        .{ .transfer_bit = true },
        .{ .host_bit = true },
        .{},
        null,
        @ptrCast(&barrier),
        null,
    );
}

pub fn readPixelsAlloc(self: *const Self, alloc: std.mem.Allocator) ![]u8 {
    const size = self.width * self.height * 4;
    const pixels = try alloc.alloc(u8, size);
    errdefer alloc.free(pixels);

    const mapped = try self.context.device.mapMemory(self.readback.memory, 0, size, .{});
    defer self.context.device.unmapMemory(self.readback.memory);
    const src: [*]const u8 = @ptrCast(mapped.?);
    @memcpy(pixels, src[0..size]);
    return pixels;
}

pub fn exportDmabuf(self: *const Self) !Dmabuf {
    const export_image = self.export_image orelse return error.DmabufUnsupported;
    const fd = try self.context.device.getMemoryFdKHR(&.{
        .memory = export_image.memory,
        .handle_type = .{ .dma_buf_bit_ext = true },
    });
    errdefer {
        if (fd >= 0) _ = std.posix.system.close(fd);
    }

    var planes: Dmabuf.Planes = .{ .count = 1 };
    planes.fds[0] = fd;
    planes.offsets[0] = @intCast(export_image.layout.offset);
    planes.strides[0] = @intCast(export_image.layout.row_pitch);
    try planes.validate();

    return .{
        .width = @intCast(self.width),
        .height = @intCast(self.height),
        .fourcc = drm_format_abgr8888,
        .modifier = export_image.modifier,
        .premultiplied = true,
        .planes = planes,
    };
}

const ExportImage = struct {
    context: *Context,
    image: vk.Image,
    memory: vk.DeviceMemory,
    modifier: u64,
    layout: vk.SubresourceLayout,

    fn init(context: *Context, width: usize, height: usize, format: vk.Format) !ExportImage {
        const modifier = try chooseModifier(context, format);

        var modifier_info: vk.ImageDrmFormatModifierListCreateInfoEXT = .{
            .drm_format_modifier_count = 1,
            .p_drm_format_modifiers = @ptrCast(&modifier),
        };
        var external_info: vk.ExternalMemoryImageCreateInfo = .{
            .p_next = &modifier_info,
            .handle_types = .{ .dma_buf_bit_ext = true },
        };
        const image = try context.device.createImage(&.{
            .p_next = &external_info,
            .image_type = .@"2d",
            .format = format,
            .extent = .{ .width = @intCast(width), .height = @intCast(height), .depth = 1 },
            .mip_levels = 1,
            .array_layers = 1,
            .samples = .{ .@"1_bit" = true },
            .tiling = .drm_format_modifier_ext,
            .usage = .{ .transfer_dst_bit = true },
            .sharing_mode = .exclusive,
            .initial_layout = .undefined,
        }, null);
        errdefer context.device.destroyImage(image, null);

        const requirements = context.device.getImageMemoryRequirements(image);
        var dedicated: vk.MemoryDedicatedAllocateInfo = .{ .image = image };
        var export_info: vk.ExportMemoryAllocateInfo = .{
            .p_next = &dedicated,
            .handle_types = .{ .dma_buf_bit_ext = true },
        };
        const memory = try context.device.allocateMemory(&.{
            .p_next = &export_info,
            .allocation_size = requirements.size,
            .memory_type_index = try context.memoryType(
                requirements.memory_type_bits,
                .{ .device_local_bit = true },
            ),
        }, null);
        errdefer context.device.freeMemory(memory, null);
        try context.device.bindImageMemory(image, memory, 0);

        var modifier_props: vk.ImageDrmFormatModifierPropertiesEXT = .{
            .drm_format_modifier = 0,
        };
        try context.device.getImageDrmFormatModifierPropertiesEXT(image, &modifier_props);

        const subresource: vk.ImageSubresource = .{
            .aspect_mask = .{ .color_bit = true },
            .mip_level = 0,
            .array_layer = 0,
        };
        const layout = context.device.getImageSubresourceLayout(image, &subresource);

        const command_buffer = try context.beginCommands();
        Texture.imageBarrier(
            context,
            command_buffer,
            image,
            .{},
            .{ .transfer_write_bit = true },
            .{ .top_of_pipe_bit = true },
            .{ .transfer_bit = true },
            .undefined,
            .general,
        );
        try context.submitCommands(command_buffer);

        return .{
            .context = context,
            .image = image,
            .memory = memory,
            .modifier = modifier_props.drm_format_modifier,
            .layout = layout,
        };
    }

    fn deinit(self: ExportImage) void {
        self.context.device.destroyImage(self.image, null);
        self.context.device.freeMemory(self.memory, null);
    }

    fn chooseModifier(context: *Context, format: vk.Format) !u64 {
        // Prefer linear because GTK import support is broadest there, but use
        // any modifier that supports transfer destinations when unavailable.
        var list: vk.DrmFormatModifierPropertiesListEXT = .{};
        var props: vk.FormatProperties2 = .{
            .p_next = &list,
            .format_properties = undefined,
        };
        context.instance.getPhysicalDeviceFormatProperties2(context.physical_device, format, &props);
        if (list.drm_format_modifier_count == 0) return error.NoDrmModifier;

        const alloc = std.heap.c_allocator;
        const modifiers = try alloc.alloc(vk.DrmFormatModifierPropertiesEXT, list.drm_format_modifier_count);
        defer alloc.free(modifiers);
        list.p_drm_format_modifier_properties = modifiers.ptr;
        context.instance.getPhysicalDeviceFormatProperties2(context.physical_device, format, &props);

        var fallback: ?u64 = null;
        for (modifiers) |value| {
            if (!value.drm_format_modifier_tiling_features.transfer_dst_bit) continue;
            if (fallback == null) fallback = value.drm_format_modifier;
            if (value.drm_format_modifier == drm_format_mod_linear) return value.drm_format_modifier;
        }
        return fallback orelse error.NoTransferDrmModifier;
    }
};
