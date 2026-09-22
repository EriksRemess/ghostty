const Self = @This();

const vk = @import("api.zig").vk;
const Context = @import("Context.zig");

pub const Options = struct {
    context: *Context,
    min_filter: vk.Filter,
    mag_filter: vk.Filter,
    address_mode: vk.SamplerAddressMode,
};

context: *Context,
sampler: vk.Sampler,

pub const Error = anyerror;

pub fn init(opts: Options) Error!Self {
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
        .unnormalized_coordinates = .false,
    }, null);
    return .{ .context = opts.context, .sampler = sampler };
}

pub fn deinit(self: Self) void {
    self.context.device.destroySampler(self.sampler, null);
}
