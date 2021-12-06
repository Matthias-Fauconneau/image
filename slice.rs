/// T should be a basic type (i.e valid when casted from any data)
pub unsafe fn cast<T>(slice: &[u8]) -> &[T] {
    std::slice::from_raw_parts(slice.as_ptr() as *const T, slice.len() / std::mem::size_of::<T>())
}

