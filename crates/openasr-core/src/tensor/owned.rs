use super::{TensorError, TensorLayout, TensorViewF32, contiguous_layout};

#[derive(Debug, Clone, PartialEq)]
pub struct TensorOwnedF32 {
    pub(crate) data: Vec<f32>,
    pub(crate) layout: TensorLayout,
}

impl TensorOwnedF32 {
    pub fn contiguous(data: Vec<f32>, shape: &[usize]) -> Result<Self, TensorError> {
        let layout = contiguous_layout(shape)?;
        if data.len() != layout.element_count() {
            return Err(TensorError::DataLengthMismatch {
                expected: layout.element_count(),
                actual: data.len(),
            });
        }
        Ok(Self { data, layout })
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    pub fn layout(&self) -> &TensorLayout {
        &self.layout
    }

    pub fn view(&self) -> TensorViewF32<'_> {
        TensorViewF32 {
            data: &self.data,
            layout: self.layout.clone(),
        }
    }
}
