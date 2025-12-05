#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Integer {
    I1,
    I8,
    I16,
    I32,
    I64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum IEEEFloat {
    F16,
    F32,
    F64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AltFloat {
    BF16,
    E3M4,
    E5M2,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Tile {
    pub dims: Box<[usize]>,
    pub elem: Box<Type>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Dimension(pub i64);

impl Dimension {
    pub const DYN: Self = Self(i64::MIN);
    pub fn fixed(size: usize) -> Self {
        Self(size as i64)
    }
    pub fn dynamic() -> Self {
        Self::DYN
    }
    pub fn is_dyn(&self) -> bool {
        self.0 == i64::MIN
    }
    pub fn as_fixed(&self) -> Option<usize> {
        if self.is_dyn() {
            None
        } else {
            Some(self.0 as usize)
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Shape(pub Box<[Dimension]>);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorView {
    pub shape: Shape,
    pub strides: Shape,
    pub ty: Box<Type>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PartitionView {
    pub tile_shape: Box<[usize]>,
    pub original_view: Box<TensorView>,
    pub dimension_map: Box<[i32]>,
    pub masked: bool
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Integer(Integer),
    Float(IEEEFloat),
    AltFloat(AltFloat),
    Pointer(Box<Self>),
    Tile(Tile),
    TensorView(TensorView),
    PartitionView(PartitionView),
}