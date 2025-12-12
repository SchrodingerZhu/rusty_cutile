use std::marker::PhantomData;

use dashmap::DashSet;

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
pub struct Tile<'a> {
    pub dims: Box<[usize]>,
    pub elem: Type<'a>,
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
pub struct TensorView<'a> {
    pub shape: Shape,
    pub strides: Shape,
    pub ty: Type<'a>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PartitionView<'a> {
    pub tile_shape: Box<[usize]>,
    pub original_view: Type<'a>,
    pub dimension_map: Box<[i32]>,
    pub masked: bool
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TypeInstance<'a> {
    Integer(Integer),
    Float(IEEEFloat),
    AltFloat(AltFloat),
    Pointer(Type<'a>),
    Tile(Tile<'a>),
    TensorView(TensorView<'a>),
    PartitionView(PartitionView<'a>),
}

type InvariantLifetime<'a> = PhantomData<*mut &'a ()>;

#[derive(Copy, Clone, Debug, Eq)]
#[repr(transparent)]
pub struct Type<'a>(&'a TypeInstance<'a>, InvariantLifetime<'a>);

impl<'a> PartialEq for Type<'a> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0, other.0)
    }
}

impl<'a> std::hash::Hash for Type<'a> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.0 as *const TypeInstance).hash(state);
    }
}

struct TypeInterner<'a> {
    set: DashSet<TypeInstance<'a>>,
}