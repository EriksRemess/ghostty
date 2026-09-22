const Self = @This();

const std = @import("std");
const api = @import("api.zig");
const vk = api.vk;

const log = std.log.scoped(.vulkan);

// vulkan-zig loads commands into dispatch tables instead of exporting global
// vk* functions. The wrappers must outlive the proxies that reference them.
base: vk.BaseWrapper,
instance_wrapper: *vk.InstanceWrapper,
device_wrapper: *vk.DeviceWrapper,
instance: vk.InstanceProxy,
physical_device: vk.PhysicalDevice,
device: vk.DeviceProxy,
queue: vk.Queue,
queue_family: u32,
command_pool: vk.CommandPool,
descriptor_set_layout: vk.DescriptorSetLayout,
pipeline_layout: vk.PipelineLayout,
memory_properties: vk.PhysicalDeviceMemoryProperties,
dmabuf_enabled: std.atomic.Value(bool),
consumer_dmabuf_formats: ?[]DmabufFormat,
alloc: std.mem.Allocator,
deferred_buffers: ?*DeferredBuffer = null,

/// A format/modifier pair accepted by the presentation consumer. Vulkan can
/// export many combinations that GTK cannot import, so targets only select
/// modifiers present in this list.
pub const DmabufFormat = struct {
    fourcc: u32,
    modifier: u64,
};

pub const DeferredBuffer = struct {
    buffer: vk.Buffer,
    memory: vk.DeviceMemory,
    next: ?*DeferredBuffer = null,
};

pub fn init(alloc: std.mem.Allocator, consumer_dmabuf_formats: ?[]const DmabufFormat) !*Self {
    const self = try alloc.create(Self);
    errdefer alloc.destroy(self);

    const owned_consumer_formats = if (consumer_dmabuf_formats) |formats|
        try alloc.dupe(DmabufFormat, formats)
    else
        null;
    errdefer if (owned_consumer_formats) |formats| alloc.free(formats);

    const app_info: vk.ApplicationInfo = .{
        .p_application_name = "Ghostty",
        .application_version = vk.makeApiVersion(0, 1, 0, 0).toU32(),
        .p_engine_name = "Ghostty",
        .engine_version = vk.makeApiVersion(0, 1, 0, 0).toU32(),
        .api_version = vk.API_VERSION_1_3.toU32(),
    };
    const instance_info: vk.InstanceCreateInfo = .{ .p_application_info = &app_info };

    const base = vk.BaseWrapper.load(api.vkGetInstanceProcAddr);
    const instance_wrapper = try alloc.create(vk.InstanceWrapper);
    errdefer alloc.destroy(instance_wrapper);
    const instance_handle = try base.createInstance(&instance_info, null);
    instance_wrapper.* = .load(instance_handle, base.dispatch.vkGetInstanceProcAddr.?);
    const instance = vk.InstanceProxy.init(instance_handle, instance_wrapper);
    errdefer instance.destroyInstance(null);

    const physical_devices = try instance.enumeratePhysicalDevicesAlloc(alloc);
    defer alloc.free(physical_devices);
    if (physical_devices.len == 0) return error.NoVulkanDevice;

    var physical_device = vk.PhysicalDevice.null_handle;
    var queue_family: u32 = 0;
    var fallback_device = vk.PhysicalDevice.null_handle;
    var fallback_family: u32 = 0;

    for (physical_devices) |candidate| {
        const props = instance.getPhysicalDeviceProperties(candidate);
        if (props.api_version < vk.API_VERSION_1_3.toU32()) continue;

        // Dynamic rendering is optional even when the implementation exposes
        // Vulkan 1.3, so reject devices that cannot provide the feature before
        // applying the discrete-GPU preference.
        var dynamic_rendering_support: vk.PhysicalDeviceDynamicRenderingFeatures = .{};
        var features: vk.PhysicalDeviceFeatures2 = .{
            .p_next = &dynamic_rendering_support,
            .features = .{},
        };
        instance.getPhysicalDeviceFeatures2(candidate, &features);
        if (dynamic_rendering_support.dynamic_rendering != .true) continue;

        const family = findGraphicsQueue(instance, alloc, candidate) catch continue;

        if (fallback_device == .null_handle) {
            fallback_device = candidate;
            fallback_family = family;
        }
        if (props.device_type == .discrete_gpu) {
            physical_device = candidate;
            queue_family = family;
            break;
        }
    }
    if (physical_device == .null_handle) {
        physical_device = fallback_device;
        queue_family = fallback_family;
    }
    if (physical_device == .null_handle) return error.NoGraphicsQueue;

    const props = instance.getPhysicalDeviceProperties(physical_device);
    const version: vk.Version = @bitCast(props.api_version);
    log.info("device={s} api={d}.{d}.{d}", .{
        std.mem.sliceTo(&props.device_name, 0), version.major, version.minor, version.patch,
    });
    if (props.api_version < vk.API_VERSION_1_3.toU32()) return error.VulkanVersionTooOld;

    const extension_support = try queryDeviceExtensions(instance, alloc, physical_device);
    var supports_dmabuf = extension_support.external_memory_fd and
        extension_support.external_memory_dma_buf and
        extension_support.image_drm_format_modifier;

    const priority: f32 = 1.0;
    const queue_info: vk.DeviceQueueCreateInfo = .{
        .queue_family_index = queue_family,
        .queue_count = 1,
        .p_queue_priorities = @ptrCast(&priority),
    };
    var dynamic_rendering: vk.PhysicalDeviceDynamicRenderingFeatures = .{
        .dynamic_rendering = .true,
    };
    const dmabuf_extensions = [_][*:0]const u8{
        vk.extensions.khr_external_memory_fd.name,
        vk.extensions.ext_external_memory_dma_buf.name,
        vk.extensions.ext_image_drm_format_modifier.name,
    };
    const device_info: vk.DeviceCreateInfo = .{
        .p_next = &dynamic_rendering,
        .queue_create_info_count = 1,
        .p_queue_create_infos = @ptrCast(&queue_info),
        .enabled_extension_count = if (supports_dmabuf) dmabuf_extensions.len else 0,
        .pp_enabled_extension_names = if (supports_dmabuf) &dmabuf_extensions else null,
    };

    const device_wrapper = try alloc.create(vk.DeviceWrapper);
    errdefer alloc.destroy(device_wrapper);
    const device_handle = try instance.createDevice(physical_device, &device_info, null);
    device_wrapper.* = .load(device_handle, instance_wrapper.dispatch.vkGetDeviceProcAddr.?);
    const device = vk.DeviceProxy.init(device_handle, device_wrapper);
    errdefer device.destroyDevice(null);

    // Extension enumeration alone is insufficient: loaders may still omit an
    // entry point. In that case Target uses its CPU presentation path.
    if (supports_dmabuf and
        (device_wrapper.dispatch.vkGetMemoryFdKHR == null or
            device_wrapper.dispatch.vkGetImageDrmFormatModifierPropertiesEXT == null))
    {
        log.warn("DMA-BUF extensions are present but entry points are unavailable; using memory presentation", .{});
        supports_dmabuf = false;
    }
    log.info("DMA-BUF export={}", .{supports_dmabuf});

    const command_pool = try device.createCommandPool(&.{
        .flags = .{ .reset_command_buffer_bit = true },
        .queue_family_index = queue_family,
    }, null);
    errdefer device.destroyCommandPool(command_pool, null);

    const all_graphics: vk.ShaderStageFlags = .{
        .vertex_bit = true,
        .tessellation_control_bit = true,
        .tessellation_evaluation_bit = true,
        .geometry_bit = true,
        .fragment_bit = true,
    };
    const bindings = [_]vk.DescriptorSetLayoutBinding{
        descriptorBinding(0, .uniform_buffer, all_graphics),
        descriptorBinding(1, .storage_buffer, all_graphics),
        descriptorBinding(2, .combined_image_sampler, all_graphics),
        descriptorBinding(3, .combined_image_sampler, all_graphics),
    };
    const descriptor_set_layout = try device.createDescriptorSetLayout(&.{
        .binding_count = bindings.len,
        .p_bindings = &bindings,
    }, null);
    errdefer device.destroyDescriptorSetLayout(descriptor_set_layout, null);

    const pipeline_layout = try device.createPipelineLayout(&.{
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast(&descriptor_set_layout),
    }, null);
    errdefer device.destroyPipelineLayout(pipeline_layout, null);

    self.* = .{
        .base = base,
        .instance_wrapper = instance_wrapper,
        .device_wrapper = device_wrapper,
        .instance = instance,
        .physical_device = physical_device,
        .device = device,
        .queue = device.getDeviceQueue(queue_family, 0),
        .queue_family = queue_family,
        .command_pool = command_pool,
        .descriptor_set_layout = descriptor_set_layout,
        .pipeline_layout = pipeline_layout,
        .memory_properties = instance.getPhysicalDeviceMemoryProperties(physical_device),
        .dmabuf_enabled = .init(supports_dmabuf),
        .consumer_dmabuf_formats = owned_consumer_formats,
        .alloc = alloc,
    };
    return self;
}

pub fn deinit(self: *Self, alloc: std.mem.Allocator) void {
    self.device.deviceWaitIdle() catch {};
    self.collectDeferredBuffers();
    self.device.destroyPipelineLayout(self.pipeline_layout, null);
    self.device.destroyDescriptorSetLayout(self.descriptor_set_layout, null);
    self.device.destroyCommandPool(self.command_pool, null);
    self.device.destroyDevice(null);
    self.instance.destroyInstance(null);
    if (self.consumer_dmabuf_formats) |formats| alloc.free(formats);
    alloc.destroy(self.device_wrapper);
    alloc.destroy(self.instance_wrapper);
    alloc.destroy(self);
}

/// Whether new frames should use the DMA-BUF presentation path. GTK may
/// disable this after an import failure, in which case every target switches
/// to its CPU readback buffer on the next frame.
pub fn dmabufEnabled(self: *const Self) bool {
    return self.dmabuf_enabled.load(.seq_cst);
}

/// Disable DMA-BUF presentation and return whether this call changed state.
pub fn disableDmabuf(self: *Self) bool {
    return self.dmabuf_enabled.swap(false, .seq_cst);
}

pub fn consumerSupportsDmabuf(self: *const Self, fourcc: u32, modifier: u64) bool {
    return consumerSupportsDmabufFormats(self.consumer_dmabuf_formats, fourcc, modifier);
}

fn consumerSupportsDmabufFormats(formats_: ?[]const DmabufFormat, fourcc: u32, modifier: u64) bool {
    const formats = formats_ orelse return true;
    for (formats) |format| {
        if (format.fourcc == fourcc and format.modifier == modifier) return true;
    }
    return false;
}

pub fn memoryType(self: *const Self, bits: u32, flags: vk.MemoryPropertyFlags) !u32 {
    var i: u32 = 0;
    while (i < self.memory_properties.memory_type_count) : (i += 1) {
        if ((bits & (@as(u32, 1) << @intCast(i))) != 0 and
            self.memory_properties.memory_types[i].property_flags.contains(flags)) return i;
    }
    return error.NoSuitableMemoryType;
}

pub fn beginCommands(self: *Self) !vk.CommandBuffer {
    var command_buffer: vk.CommandBuffer = undefined;
    try self.device.allocateCommandBuffers(&.{
        .command_pool = self.command_pool,
        .level = .primary,
        .command_buffer_count = 1,
    }, @ptrCast(&command_buffer));
    errdefer self.device.freeCommandBuffers(self.command_pool, @ptrCast(&command_buffer));
    try self.device.beginCommandBuffer(command_buffer, &.{
        .flags = .{ .one_time_submit_bit = true },
    });
    return command_buffer;
}

pub fn submitCommands(self: *Self, command_buffer: vk.CommandBuffer) !void {
    errdefer self.device.freeCommandBuffers(self.command_pool, @ptrCast(&command_buffer));
    try self.device.endCommandBuffer(command_buffer);

    const fence = try self.device.createFence(&.{}, null);
    defer self.device.destroyFence(fence, null);
    const submit_info: vk.SubmitInfo = .{
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&command_buffer),
    };
    // Resource replacement defers old buffers until this fence signals. This
    // keeps handles referenced by the just-recorded frame alive on the GPU.
    // GTK's DMA-BUF texture API has no acquire-fence input, so presentation
    // must also wait here rather than exposing an image still being written.
    try self.device.queueSubmit(self.queue, @ptrCast(&submit_info), fence);
    _ = try self.device.waitForFences(@ptrCast(&fence), .true, std.math.maxInt(u64));
    self.device.freeCommandBuffers(self.command_pool, @ptrCast(&command_buffer));
    self.collectDeferredBuffers();
}

pub fn deferBuffer(self: *Self, node: *DeferredBuffer) void {
    node.next = self.deferred_buffers;
    self.deferred_buffers = node;
}

fn collectDeferredBuffers(self: *Self) void {
    var current = self.deferred_buffers;
    self.deferred_buffers = null;
    while (current) |node| {
        current = node.next;
        self.device.destroyBuffer(node.buffer, null);
        self.device.freeMemory(node.memory, null);
        self.alloc.destroy(node);
    }
}

fn descriptorBinding(binding: u32, descriptor_type: vk.DescriptorType, stages: vk.ShaderStageFlags) vk.DescriptorSetLayoutBinding {
    return .{
        .binding = binding,
        .descriptor_type = descriptor_type,
        .descriptor_count = 1,
        .stage_flags = stages,
    };
}

fn findGraphicsQueue(instance: vk.InstanceProxy, alloc: std.mem.Allocator, device: vk.PhysicalDevice) !u32 {
    const props = try instance.getPhysicalDeviceQueueFamilyPropertiesAlloc(device, alloc);
    defer alloc.free(props);
    for (props, 0..) |prop, i| if (prop.queue_flags.graphics_bit) return @intCast(i);
    return error.NoGraphicsQueue;
}

const ExtensionSupport = struct {
    external_memory_fd: bool = false,
    external_memory_dma_buf: bool = false,
    image_drm_format_modifier: bool = false,
};

fn queryDeviceExtensions(instance: vk.InstanceProxy, alloc: std.mem.Allocator, device: vk.PhysicalDevice) !ExtensionSupport {
    const props = try instance.enumerateDeviceExtensionPropertiesAlloc(device, null, alloc);
    defer alloc.free(props);

    var support: ExtensionSupport = .{};
    for (props) |prop| {
        const name = std.mem.sliceTo(&prop.extension_name, 0);
        if (std.mem.eql(u8, name, vk.extensions.khr_external_memory_fd.name)) support.external_memory_fd = true;
        if (std.mem.eql(u8, name, vk.extensions.ext_external_memory_dma_buf.name)) support.external_memory_dma_buf = true;
        if (std.mem.eql(u8, name, vk.extensions.ext_image_drm_format_modifier.name)) support.image_drm_format_modifier = true;
    }
    return support;
}

test "DMA-BUF consumer formats require an exact pair" {
    const formats = [_]DmabufFormat{
        .{ .fourcc = 1, .modifier = 10 },
        .{ .fourcc = 2, .modifier = 20 },
    };

    try std.testing.expect(consumerSupportsDmabufFormats(&formats, 1, 10));
    try std.testing.expect(!consumerSupportsDmabufFormats(&formats, 1, 20));
    try std.testing.expect(!consumerSupportsDmabufFormats(&formats, 3, 10));
    try std.testing.expect(consumerSupportsDmabufFormats(null, 3, 30));
}
