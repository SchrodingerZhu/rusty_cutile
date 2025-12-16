use std::cell::RefCell;
use std::hash::BuildHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::marker::PhantomData;

use hashbrown::hash_table::Entry;

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
    pub masked: bool,
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

pub struct TypeInterner<'a> {
    ctx_token: InvariantLifetime<'a>,
    bump: &'a bumpalo::Bump,
    storage: RefCell<hashbrown::HashTable<&'a TypeInstance<'a>>>,
    hasher: rustc_hash::FxRandomState,
}

macro_rules! impl_singleton_types {
    ($($name:ident => $variant:expr),* $(,)?) => {
        $(
            pub fn $name(&self) -> Type<'a> {
                self.intern($variant)
            }
        )*
    }
}

impl<'a> TypeInterner<'a> {
    // # Safety
    // One should only create one interner per context.
    pub unsafe fn new(token: InvariantLifetime<'a>, bump: &'a bumpalo::Bump) -> Self {
        Self {
            ctx_token: token,
            bump,
            storage: RefCell::new(hashbrown::HashTable::new()),
            hasher: rustc_hash::FxRandomState::default(),
        }
    }

    fn intern(&self, ty: TypeInstance<'a>) -> Type<'a> {
        let mut storage = self.storage.borrow_mut();
        let hasher = |x: &&TypeInstance<'a>| {
            let mut hasher = self.hasher.build_hasher();
            x.hash(&mut hasher);
            hasher.finish()
        };
        let hash = hasher(&&ty);
        match storage.entry(hash, |x| *x == &ty, hasher) {
            Entry::Occupied(o) => Type(o.get(), self.ctx_token),
            Entry::Vacant(v) => {
                let ty_ref: &'a TypeInstance<'a> = self.bump.alloc(ty);
                v.insert(ty_ref);
                Type(ty_ref, self.ctx_token)
            }
        }
    }

    impl_singleton_types! {
        i1 => TypeInstance::Integer(Integer::I1),
        i8 => TypeInstance::Integer(Integer::I8),
        i16 => TypeInstance::Integer(Integer::I16),
        i32 => TypeInstance::Integer(Integer::I32),
        i64 => TypeInstance::Integer(Integer::I64),
        f16 => TypeInstance::Float(IEEEFloat::F16),
        f32 => TypeInstance::Float(IEEEFloat::F32),
        f64 => TypeInstance::Float(IEEEFloat::F64),
        bf16 => TypeInstance::AltFloat(AltFloat::BF16),
        e3m4 => TypeInstance::AltFloat(AltFloat::E3M4),
        e5m2 => TypeInstance::AltFloat(AltFloat::E5M2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_bump_and_token<F, R>(f: F) -> R
    where
        F: for<'a> FnOnce(&'a bumpalo::Bump, InvariantLifetime<'a>) -> R,
    {
        let bump = bumpalo::Bump::new();
        let ctx = PhantomData;
        f(&bump, ctx)
    }

    #[test]
    fn test_interning() {
        with_bump_and_token(|bump, token| {
            let interner = unsafe { TypeInterner::new(token, bump) };
            let t1 = interner.i32();
            let t2 = interner.i32();
            assert_eq!(t1, t2);
        });
    }
}
