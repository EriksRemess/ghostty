const std = @import("std");
const vk = @import("api.zig").vk;
const Context = @import("Context.zig");

pub const Options = struct {
    context: *Context,
    usage: vk.BufferUsageFlags,
};

pub const Handle = struct {
    context: *Context,
    buffer: vk.Buffer,
    memory: vk.DeviceMemory,
    size: usize,
    deferred: *Context.DeferredBuffer,

    pub fn init(opts: Options, size_: usize) !Handle {
        const size = @max(size_, 1);
        const info: vk.BufferCreateInfo = .{
            .size = size,
            .usage = opts.usage,
            .sharing_mode = .exclusive,
        };
        const buffer = try opts.context.device.createBuffer(&info, null);
        errdefer opts.context.device.destroyBuffer(buffer, null);

        const requirements = opts.context.device.getBufferMemoryRequirements(buffer);
        const alloc_info: vk.MemoryAllocateInfo = .{
            .allocation_size = requirements.size,
            .memory_type_index = try opts.context.memoryType(
                requirements.memory_type_bits,
                .{ .host_visible_bit = true, .host_coherent_bit = true },
            ),
        };

        const memory = try opts.context.device.allocateMemory(&alloc_info, null);
        errdefer opts.context.device.freeMemory(memory, null);
        try opts.context.device.bindBufferMemory(buffer, memory, 0);

        const deferred = try opts.context.alloc.create(Context.DeferredBuffer);
        errdefer opts.context.alloc.destroy(deferred);
        deferred.* = .{ .buffer = buffer, .memory = memory };

        return .{
            .context = opts.context,
            .buffer = buffer,
            .memory = memory,
            .size = size,
            .deferred = deferred,
        };
    }

    pub fn deinit(self: Handle) void {
        // Buffers may still be referenced by the command buffer currently
        // being recorded. Context frees them after the next submission fence.
        self.context.deferBuffer(self.deferred);
    }

    pub fn write(self: Handle, offset: usize, bytes: []const u8) !void {
        if (offset + bytes.len > self.size) return error.BufferOverflow;
        const mapped = try self.context.device.mapMemory(
            self.memory,
            offset,
            bytes.len,
            .{},
        );
        defer self.context.device.unmapMemory(self.memory);
        const dst: [*]u8 = @ptrCast(mapped.?);
        @memcpy(dst[0..bytes.len], bytes);
    }
};

pub fn Buffer(comptime T: type) type {
    return struct {
        const Self = @This();

        opts: Options,
        buffer: Handle,
        len: usize,

        pub fn init(opts: Options, len: usize) !Self {
            return .{
                .opts = opts,
                .buffer = try .init(opts, len * @sizeOf(T)),
                .len = len,
            };
        }

        pub fn initFill(opts: Options, data: []const T) !Self {
            var self: Self = .{
                .opts = opts,
                .buffer = try .init(opts, data.len * @sizeOf(T)),
                .len = data.len,
            };
            errdefer self.buffer.deinit();
            try self.buffer.write(0, std.mem.sliceAsBytes(data));
            return self;
        }

        pub fn deinit(self: Self) void {
            self.buffer.deinit();
        }

        pub fn sync(self: *Self, data: []const T) !void {
            if (data.len > self.len) {
                const new_len = @max(data.len * 2, 1);
                const replacement = try Handle.init(self.opts, new_len * @sizeOf(T));
                self.buffer.deinit();
                self.buffer = replacement;
                self.len = new_len;
            }
            try self.buffer.write(0, std.mem.sliceAsBytes(data));
        }

        pub fn syncFromArrayLists(self: *Self, lists: []const std.ArrayListUnmanaged(T)) !usize {
            var total_len: usize = 0;
            for (lists) |list| total_len += list.items.len;
            if (total_len > self.len) {
                const new_len = @max(total_len * 2, 1);
                const replacement = try Handle.init(self.opts, new_len * @sizeOf(T));
                self.buffer.deinit();
                self.buffer = replacement;
                self.len = new_len;
            }

            var offset: usize = 0;
            for (lists) |list| {
                const bytes = std.mem.sliceAsBytes(list.items);
                try self.buffer.write(offset, bytes);
                offset += bytes.len;
            }
            return total_len;
        }
    };
}
