const Self = @This();

const std = @import("std");
const vk = @import("api.zig").vk;
const Context = @import("Context.zig");
const RenderPass = @import("RenderPass.zig");
const Target = @import("Target.zig");
const Vulkan = @import("../Vulkan.zig");
const Renderer = @import("../generic.zig").Renderer(Vulkan);
const Health = @import("../../renderer.zig").Health;

const log = std.log.scoped(.vulkan);

pub const Options = struct {};

context: *Context,
renderer: *Renderer,
target: *Target,
command_buffer: vk.CommandBuffer,
descriptor_pool: vk.DescriptorPool,

pub fn begin(opts: Options, renderer: *Renderer, target: *Target) !Self {
    _ = opts;
    const context = renderer.api.context;
    const command_buffer = try context.beginCommands();
    errdefer context.device.freeCommandBuffers(context.command_pool, @ptrCast(&command_buffer));

    // Descriptor sets are frame-local, so destroying this pool after submit
    // releases every set allocated while recording the frame at once.
    const max_sets = 4096;
    const pool_sizes = [_]vk.DescriptorPoolSize{
        .{ .type = .uniform_buffer, .descriptor_count = max_sets },
        .{ .type = .storage_buffer, .descriptor_count = max_sets },
        .{ .type = .combined_image_sampler, .descriptor_count = max_sets * 2 },
    };
    const descriptor_pool = try context.device.createDescriptorPool(&.{
        .max_sets = max_sets,
        .pool_size_count = pool_sizes.len,
        .p_pool_sizes = &pool_sizes,
    }, null);

    return .{
        .context = context,
        .renderer = renderer,
        .target = target,
        .command_buffer = command_buffer,
        .descriptor_pool = descriptor_pool,
    };
}

pub inline fn renderPass(self: *const Self, attachments: []const RenderPass.Options.Attachment) RenderPass {
    return .begin(self.context, self.command_buffer, self.descriptor_pool, .{ .attachments = attachments });
}

pub fn complete(self: *Self, sync: bool) void {
    _ = sync;
    const presentation = self.target.recordPresentation(self.command_buffer);
    self.context.submitCommands(self.command_buffer) catch |err| {
        log.warn("failed to submit frame err={}", .{err});
        self.context.device.destroyDescriptorPool(self.descriptor_pool, null);
        self.renderer.frameCompleted(.unhealthy);
        return;
    };
    self.context.device.destroyDescriptorPool(self.descriptor_pool, null);

    const frame = self.renderer.api.present(self.target.*, presentation) catch |err| {
        log.warn("failed to present frame err={}", .{err});
        self.renderer.frameCompleted(.unhealthy);
        return;
    };
    self.renderer.pushFrame(frame);
    _ = self.renderer.surface_mailbox.push(.redraw, .{ .forever = {} });
    self.renderer.frameCompleted(Health.healthy);
}
