pub const vk = @import("vulkan");

pub extern fn vkGetInstanceProcAddr(
    instance: vk.Instance,
    name: [*:0]const u8,
) callconv(.c) vk.PfnVoidFunction;
