#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityClass {
    Cpu,
    Gpu,
    Npu,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCapabilities {
    pub classes: Vec<CapabilityClass>,
    pub supports_fp16: bool,
    pub supports_int8: bool,
}
