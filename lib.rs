#![feature(once_cell,type_alias_impl_trait,slice_take,new_uninit,const_trait_impl)]
#![allow(non_upper_case_globals)]
pub use vector::xy;
use vector::{size, uint2, Rect, Lerp};

pub struct Image<D> {
	pub data : D,
	pub size : size,
	pub stride : u32,
}

impl<D> std::fmt::Debug for Image<D> { fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(),std::fmt::Error> { assert!(self.stride == self.size.x); write!(f, "{:?}", self.size) } }

impl<D> Image<D> {
	pub fn index(&self, xy{x,y}: uint2) -> usize { assert!(x < self.size.x && y < self.size.y); (y * self.stride + x) as usize }
	fn rect(&self) -> Rect { self.size.into() }

	#[track_caller] pub fn strided<T>(data: D, size : size, stride: u32) -> Self where D:AsRef<[T]> {
		assert_eq!(data.as_ref().len(), (stride*size.y) as usize);
		Self{data, size, stride}
	}
	#[track_caller] pub fn new<T>(size : size, data: D) -> Self where D:AsRef<[T]> { Self::strided(data, size, size.x) }

	pub fn as_ref<T>(&self) -> Image<&[T]> where D:AsRef<[T]> { Image{data: self.data.as_ref(), size: self.size, stride: self.stride} }
	pub fn as_mut<T>(&mut self) -> Image<&mut [T]> where D:AsMut<[T]> { Image{data: self.data.as_mut(), size:self.size, stride: self.stride} }
}

impl<D> std::ops::Deref for Image<D> {
	type Target = D;
	fn deref(&self) -> &Self::Target { &self.data }
}

impl<D> std::ops::DerefMut for Image<D> {
	fn deref_mut(&mut self) -> &mut Self::Target { &mut self.data }
}

impl<T, D:std::ops::Deref<Target=[T]>> std::ops::Index<usize> for Image<D> {
	type Output=T;
	fn index(&self, i:usize) -> &Self::Output { &self.data[i] }
}
impl<T, D:std::ops::DerefMut<Target=[T]>> std::ops::IndexMut<usize> for Image<D> {
	fn index_mut(&mut self, i:usize) -> &mut Self::Output { &mut self.data[i] }
}

impl<D> std::ops::Index<uint2> for Image<D> where Self: std::ops::Index<usize> {
	type Output = <Self as std::ops::Index<usize>>::Output;
	fn index(&self, i:uint2) -> &Self::Output { &self[self.index(i)] }
}
impl<D> std::ops::IndexMut<uint2> for Image<D> where Self: std::ops::IndexMut<usize> {
	fn index_mut(&mut self, i:uint2) -> &mut Self::Output { let i = self.index(i); &mut self[i] }
}

impl<T, D:std::ops::Deref<Target=[T]>> Image<D> {
	pub fn rows(&self, rows: std::ops::Range<u32>) -> Image<&[T]> {
		assert!(rows.end <= self.size.y);
		Image{size: xy{x: self.size.x, y: rows.len() as u32}, stride: self.stride, data: &self.data[(rows.start*self.stride) as usize..]}
	}
	#[track_caller] pub fn slice(&self, offset: uint2, size: size) -> Image<&[T]> {
		assert!(offset.x+size.x <= self.size.x && offset.y+size.y <= self.size.y, "{:?} {:?} {:?} {:?}", offset, size, self.size, offset+size);
		Image{size, stride: self.stride, data: &self.data[(offset.y*self.stride+offset.x) as usize..]}
	}
}

impl<T, D:std::ops::DerefMut<Target=[T]>> Image<D> {
	pub fn rows_mut(&mut self, rows: std::ops::Range<u32>) -> Image<&mut [T]> {
		assert!(rows.end <= self.size.y);
		Image{size: xy{x: self.size.x, y: rows.len() as u32}, stride: self.stride, data: &mut self.data[(rows.start*self.stride) as usize..]}
	}
	#[track_caller] pub fn slice_mut(&mut self, offset: uint2, size: size) -> Image<&mut[T]> {
		assert!(offset.x < self.size.x && offset.x.checked_add(size.x).unwrap() <= self.size.x && offset.y < self.size.y && offset.y.checked_add(size.y).unwrap() <= self.size.y && self.data.len() >= (offset.y*self.stride+offset.x) as usize,"{:?} {:?} {:?} {:?}", offset, size, self.size, offset+size);
		Image{size, stride: self.stride, data: &mut self.data[(offset.y*self.stride+offset.x) as usize..]}
	}
	#[track_caller] pub fn slice_mut_clip(&mut self, sub: Rect) -> Option<Image<&mut[T]>> {
		let sub = self.rect().clip(sub);
		Some(self.slice_mut(sub.min.unsigned(), Some(sub.size()).filter(|s| s.x > 0 && s.y > 0)?))
	}
}

impl<'t, T> Image<&'t [T]> {
	pub fn take<'s>(&'s mut self, mid: u32) -> Image<&'t [T]> {
		assert!(mid <= self.size.y);
		self.size.y -= mid;
		Image{size: xy{x: self.size.x, y: mid}, stride: self.stride, data: self.data.take(..(mid*self.stride) as usize).unwrap()}
	}
}

impl<'t, T> Image<&'t mut [T]> {
	pub fn take_mut<'s>(&'s mut self, mid: u32) -> Image<&'t mut[T]> {
		assert!(mid <= self.size.y);
		self.size.y -= mid;
		Image{size: xy{x: self.size.x, y: mid}, stride: self.stride, data: self.data.take_mut(..(mid*self.stride) as usize).unwrap()}
	}
}

impl<'t, T> Iterator for Image<&'t [T]> { // Row iterator
	type Item = &'t [T];
	fn next(&mut self) -> Option<Self::Item> {
		if self.size.y > 0 { Some(&self.take(1).data[..self.size.x as usize]) }
		else { None }
	}
}

impl<'t, T> Iterator for Image<&'t mut [T]> { // Row iterator
	type Item = &'t mut[T];
	fn next(&mut self) -> Option<Self::Item> {
		if self.size.y > 0 { Some(&mut self.take_mut(1).data[..self.size.x as usize]) }
		else { None }
	}
}

pub fn segment(total_length: u32, segment_count: u32) -> impl Iterator<Item=std::ops::Range<u32>> {
	(0..segment_count)
	.map(move |i| (i+1)*total_length/segment_count)
	.scan(0, |start, end| { let next = (*start, end); *start = end; Some(next) })
	.map(|(start, end)| start..end)
}

impl<T:Send> Image<&mut [T]> {
	#[track_caller] pub fn set<F:Fn(uint2)->T+Copy>(&mut self, f:F) { for y in 0..self.size.y { for x in 0..self.size.x { self[xy{x,y}] = f(xy{x,y}); } } }
	pub fn set_map<F:Fn(uint2,&T)->T+Copy+Send>(&mut self, f:F) { for y in 0..self.size.y { for x in 0..self.size.x { self[xy{x,y}] = f(xy{x,y},&self[xy{x,y}]); } } }
	pub fn map<F:Fn(&T)->T+Copy+Send>(&mut self, f:F) { for y in 0..self.size.y { for x in 0..self.size.x { self[xy{x,y}] = f(&self[xy{x,y}]); } } }
	pub fn zip<U:Send+Sync, D:std::ops::Deref<Target=[U]>+Send, F:Fn(&T,&U)->T+Copy+Send>(&mut self, source: &Image<D>, f: F) {
		assert!(self.size == source.size);
		for y in 0..self.size.y { for x in 0..self.size.x { self[xy{x,y}] = f(&self[xy{x,y}], &source[xy{x,y}]); } }
	}
}

pub fn fill<T:Copy+Send>(target: &mut Image<&mut [T]>, value: T) { target.set(|_| value) }

impl<T> Image<Box<[T]>> {
	pub fn from_iter<I:IntoIterator<Item=T>>(size : size, iter : I) -> Self { Self::new(size, iter.into_iter().take((size.y*size.x) as usize).collect()) }
	pub fn uninitialized(size: size) -> Self { Self::new(size, unsafe{Box::new_uninit_slice((size.x * size.y) as usize).assume_init()}) }
}
impl<T:Copy> Image<Box<[T]>> {
	pub fn fill(size: size, value: T) -> Self { Self::from_iter(size, std::iter::from_fn(|| Some(value))) }
}
impl<T:num::Zero> Image<Box<[T]>> {
	pub fn zero(size: size) -> Self { Self::from_iter(size, std::iter::from_fn(|| Some(num::zero()))) }
}

impl<'t, T: bytemuck::Pod> Image<&'t [T]> {
	pub fn cast_slice<U:bytemuck::Pod>(slice: &'t [U], size: size) -> Self { Self::new(size, bytemuck::cast_slice(slice)) }
}

impl<'t, T: bytemuck::Pod> Image<&'t mut [T]> {
	#[track_caller] pub fn cast_slice_mut<U:bytemuck::Pod>(slice: &'t mut [U], size: size, stride: u32) -> Self { Self::strided(bytemuck::cast_slice_mut(slice), size, stride) }
}

mod vector_bgr { vector::vector!(3 bgr T T T, b g r, Blue Green Red); } pub use vector_bgr::bgr;
mod vector_rgb { vector::vector!(3 rgb T T T, r g b, Red Green Blue); } pub use vector_rgb::rgb;

#[allow(non_camel_case_types)] pub type bgrf = bgr<f32>;
impl bgrf { pub fn clamp(&self) -> Self { Self{b: self.b.clamp(0.,1.), g: self.g.clamp(0.,1.), r: self.r.clamp(0.,1.)} } }

/*mod vector_bgra { vector::vector!(4 bgra T T T T, b g r a, Blue Green Red Alpha); } pub use vector_bgra::bgra;
#[allow(non_camel_case_types)] pub type bgra8 = bgra<u8>;
impl bgra8 {
	#[must_use] pub fn saturating_add(self, b: Self) -> Self { self.into_iter().zip(b).map(|(a,b)| a.saturating_add(b)).collect() }
	pub fn saturating_add_assign(&mut self, b: Self) { *self = self.saturating_add(b) }
}
//impl<T> From<bgr<T>> for bgra::bgra<T> { fn from(v: bgr<T>) -> Self { Self{b:v.b, g:v.g, r:v.r, a:T::MAX} } }
//impl From<bgr<u8>> for bgra::bgra<u8> { fn from(v: bgr<u8>) -> Self { Self{b:v.b, g:v.g, r:v.r, a:u8::MAX} } }
impl From<rgb<u8>> for bgra<u8> { fn from(v: rgb<u8>) -> Self { Self{b:v.b, g:v.g, r:v.r, a:u8::MAX} } }*/

impl From<u32> for bgr<u16> { fn from(bgr: u32) -> Self { bgr{b: (bgr >> 20) as u16 & 0x3FF, g: (bgr >> 10) as u16 & 0x3FF, r: (bgr >> 00) as u16 & 0x3FF} } }
impl From<bgr<u16>> for u32 { fn from(bgr{b,g,r}: bgr<u16>) -> Self { ((b as u32) << 20) | ((g as u32) << 10) | (r as u32) } }
/*pub fn multiply(target: &mut Image<&mut [u32]>, bgr{b,g,r}: bgrf, source: &Image<&[u16]>) {
	let bgr{b,g,r} = bgr{b: (b*1024.) as u32, g: (g*1024.) as u32, r: (r*1024.) as u32};
	target.set_map(source, |_,&source| {
		let s = source as u32;
		let bgr{b,g,r} = bgr{b: (s*b)>>10, g: (s*g)>>10, r: (s*r)>>10};
		(b<<20) | (g<<10) | r
	})
}*/
pub fn invert(image: &mut Image<&mut [u32]>, m: bgr<bool>) {
	image.map(|&bgr| { let bgr{b,g,r} = bgr::<u16>::from(bgr); bgr{b: if m.b {0x3FF-b} else {b}, g: if m.g {0x3FF-g} else {g}, r: if m.r {0x3FF-r} else {r}}.into()})
}

#[allow(non_snake_case)] fn PQ_inverse_EOTF(y: f32) -> f32 {
	#![allow(non_upper_case_globals)]
	const m1 : f32 = 2610./16384.;
	const m2 : f32 = 128.*2523./4096.;
	const c2 : f32 = 32.*2413./4096.;
	const c3 : f32 = 32.*2392./4096.;
	const c1 : f32 = c3-c2+1.;
	((c1+c2*y.powf(m1))/(1.+c3*y.powf(m1))).powf(m2)
}
#[allow(non_snake_case)] pub fn do_PQ10(v: f32) -> u16 {
	assert!(v >= 0. && v <= 1.);
	let PQ = PQ_inverse_EOTF((508./10000./*nit*/)*v);
	assert!(PQ <= 1.);
	(1023.*PQ).round() as u16
}

use std::sync::LazyLock;
static PQ10_forward14 : LazyLock<[u16; 0x4000]> = LazyLock::new(|| std::array::from_fn(|i|do_PQ10(i as f32 / 0x3FFF as f32)));
#[allow(non_snake_case)] pub fn PQ10(v: f32) -> u16 { PQ10_forward14[(0x3FFF as f32*v) as usize] } // 32K (fixme: interpolation of a smaller table might be more efficient)

impl From<bgrf> for u32 { fn from(bgr: bgrf) -> Self { bgr.map(|&c| PQ10(c)).into() } }
#[allow(non_snake_case)] pub fn PQ10_from_linear(linear : &Image<&[f32]>) -> Image<Box<[u16]>> { Image::from_iter(linear.size, linear.data.iter().map(|&v| PQ10(v))) }

#[allow(non_snake_case)] pub fn PQ10_inverse(v: u16) -> f32 { v as f32/0x3FF as f32 } // TODO
impl From<u32> for bgrf { fn from(bgr: u32) -> Self { bgr::from(bgr).map(|&c| PQ10_inverse(c)) } }

pub fn lerp(t: f32, a: u32, b: bgrf) -> u32 { u32::/*PQ10*/from(t.lerp(bgrf::/*PQ10⁻¹*/from(a), b)) }
pub fn blend(mask : &Image<&[f32]>, target: &mut Image<&mut [u32]>, color: bgrf) { target.zip(mask, |&target, &t| lerp(t, target, color)); }

#[allow(non_upper_case_globals)] const sRGB_to_PQ10 : LazyLock<[u16; 256]> = LazyLock::new(|| std::array::from_fn(|i| {
    let x = i as f32 / 255.;
    do_PQ10(if x > 0.04045 { f32::powf((x+0.055)/1.055, 2.4) } else { x / 12.92 })
}));
#[allow(non_camel_case_types)] pub type rgb8 = rgb<u8>;
impl From<rgb8> for u32 { fn from(rgb{r,g,b}: rgb8) -> Self { bgr{b: sRGB_to_PQ10[b as usize], g: sRGB_to_PQ10[g as usize], r: sRGB_to_PQ10[r as usize]}.into() } }

/*#[allow(non_snake_case)] fn sRGB(linear: f64) -> f64 { if linear > 0.0031308 {1.055*linear.powf(1./2.4)-0.055} else {12.92*linear} }
static sRGB_forward12 : LazyLock<[u8; 0x1000]> = LazyLock::new(|| std::array::from_fn(|i|(0xFF as f64 * sRGB(i as f64 / 0xFFF as f64)).round() as u8));
#[allow(non_snake_case)] pub fn sRGB8(v: f32) -> u8 { sRGB_forward12[(0xFFF as f32*v) as usize] } // 4K (fixme: interpolation of a smaller table might be faster)
impl From<bgrf> for bgra8 { fn from(bgr{b,g,r}: bgrf) -> Self { Self{b:sRGB8(b), g:sRGB8(g), r:sRGB8(r), a:0xFF} } }
#[allow(non_snake_case)] pub fn from_linear(linear : &Image<&[f32]>) -> Image<Box<[u8]>> { Image::from_iter(linear.size, linear.data.iter().map(|&v| sRGB8(v))) }*/
