pub mod core;
pub mod ty;

#[repr(transparent)]
pub struct Value(pub(crate) usize);
