#![cfg_attr(feature="type_alias_impl_trait",feature(type_alias_impl_trait))]
#![cfg_attr(feature="slice_take",feature(slice_take))]
#![cfg_attr(feature="new_uninit",feature(new_uninit))]
#![cfg_attr(feature="const_trait_impl",feature(const_trait_impl))]

#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]
pub use vector::{xy, size};
use vector::{uint2, Rect};

pub struct Image<D> {
	pub data : D,
	pub size : size,
	pub stride : u32,
}

impl<D> Image<D> {
	//#[track_caller] pub fn index(&self, xy{x,y}: uint2) -> usize { assert!(x < self.size.x && y < self.size.y); (y * self.stride + x) as usize }
	#[track_caller] pub fn index(&self, xy{x,y}: uint2) -> Option<usize> { (x < self.size.x && y < self.size.y).then(|| (y * self.stride + x) as usize) }
	pub fn rect(&self) -> Rect { self.size.into() }

	#[track_caller] pub fn strided<T>(data: D, size : size, stride: u32) -> Self where D:AsRef<[T]> {
		assert!((stride*(size.y-1)+size.x) as usize <= data.as_ref().len(), "{} {} {}", data.as_ref().len(), size, stride);
		Self{data, size, stride}
	}
	#[track_caller] pub fn new<T>(size : size, data: D) -> Self where D:AsRef<[T]> { Self::strided(data, size, size.x) }

	pub fn as_ref<T>(&self) -> Image<&[T]> where D:AsRef<[T]> { Image{data: self.data.as_ref(), size: self.size, stride: self.stride} }
	pub fn as_mut<T>(&mut self) -> Image<&mut [T]> where D:AsMut<[T]> { Image{data: self.data.as_mut(), size:self.size, stride: self.stride} }
}

use std::ops::{Deref, DerefMut, Index, IndexMut, Range};
/*impl<D> Deref for Image<D> {
	type Target = D;
	fn deref(&self) -> &Self::Target { &self.data }
}

impl<D> DerefMut for Image<D> {
	fn deref_mut(&mut self) -> &mut Self::Target { &mut self.data }
}*/

impl<T, D:Deref<Target=[T]>> Index<usize> for Image<D> {
	type Output=T;
	fn index(&self, i: usize) -> &Self::Output { &self.data[i] }
}
impl<T, D:DerefMut<Target=[T]>> IndexMut<usize> for Image<D> {
	fn index_mut(&mut self, i: usize) -> &mut Self::Output { &mut self.data[i] }
}

impl<D> Index<uint2> for Image<D> where Self: Index<usize> {
	type Output = <Self as Index<usize>>::Output;
	fn index(&self, i: uint2) -> &Self::Output { &self[self.index(i).unwrap()] }
}
impl<D> IndexMut<uint2> for Image<D> where Self: IndexMut<usize> {
	fn index_mut(&mut self, i: uint2) -> &mut Self::Output { let i = self.index(i).unwrap(); &mut self[i] }
}

impl<T, D:Deref<Target=[T]>> Image<D> {
	pub fn get(&self, i: uint2) -> Option<&T> { self.index(i).map(|i| &self[i]) }
	pub fn rows(&self, rows: Range<u32>) -> Image<&[T]> {
		assert!(rows.end <= self.size.y);
		Image{size: xy{x: self.size.x, y: rows.len() as u32}, stride: self.stride, data: &self.data[(rows.start*self.stride) as usize..]}
	}
	#[track_caller] pub fn slice(&self, offset: uint2, size: size) -> Image<&[T]> {
		assert!(offset.x+size.x <= self.size.x && offset.y+size.y <= self.size.y, "{:?} {:?} {:?} {:?}", offset, size, self.size, offset+size);
		let start = offset.y*self.stride+offset.x;
		//Image{size, stride: self.stride, data: &self.data[start as usize..(start+(size.y-1)*self.stride+size.x) as usize]}
		Image{size, stride: self.stride, data: &self.data[start as usize..(start+size.y*self.stride) as usize]}
	}
}

impl<T, D:DerefMut<Target=[T]>> Image<D> {
	pub fn get_mut(&mut self, i: uint2) -> Option<&mut T> { self.index(i).map(|i| &mut self[i]) }
	pub fn rows_mut(&mut self, rows: Range<u32>) -> Image<&mut [T]> {
		assert!(rows.end <= self.size.y);
		Image{size: xy{x: self.size.x, y: rows.len() as u32}, stride: self.stride, data: &mut self.data[(rows.start*self.stride) as usize..]}
	}
	#[track_caller] pub fn slice_mut(&mut self, offset: uint2, size: size) -> Image<&mut[T]> {
		assert!(offset.x < self.size.x && offset.x.checked_add(size.x).unwrap() <= self.size.x && offset.y < self.size.y && offset.y.checked_add(size.y).unwrap() <= self.size.y && self.data.len() >= (offset.y*self.stride+offset.x) as usize,"{:?} {:?} {:?} {:?}", offset, size, self.size, offset+size);
		Image{size, stride: self.stride, data: &mut self.data[(offset.y*self.stride+offset.x) as usize..]}
	}
	#[track_caller] pub fn slice_mut_clip(&mut self, sub: Rect) -> Option<Image<&mut[T]>> {
		let sub = self.rect().clip(sub);
		Some(self.slice_mut(sub.min.unsigned(), Some(sub.size().unsigned()).filter(|s| s.x > 0 && s.y > 0)?))
	}
}

#[cfg(feature="slice_take")] impl<'t, T> Image<&'t [T]> {
	#[track_caller] pub fn take<'s>(&'s mut self, mid: u32) -> Image<&'t [T]> {
		assert!(mid <= self.size.y);
		self.size.y -= mid;
		Image{size: xy{x: self.size.x, y: mid}, stride: self.stride, data: self.data.take(..(mid*self.stride) as usize).unwrap()}
	}
}

#[cfg(feature="slice_take")] impl<'t, T> Image<&'t mut [T]> {
	pub fn take_mut<'s>(&'s mut self, mid: u32) -> Image<&'t mut[T]> {
		assert!(mid <= self.size.y);
		self.size.y -= mid;
		Image{size: xy{x: self.size.x, y: mid}, stride: self.stride, data: self.data.take_mut(..(mid*self.stride) as usize).unwrap()}
	}
}

#[cfg(feature="slice_take")] impl<'t, T> Iterator for Image<&'t [T]> { // Row iterator
	type Item = &'t [T];
	fn next(&mut self) -> Option<Self::Item> {
		if self.size.y > 0 { Some(&self.take(1).data[..self.size.x as usize]) }
		else { None }
	}
}

#[cfg(feature="slice_take")] impl<'t, T> Iterator for Image<&'t mut [T]> { // Row iterator
	type Item = &'t mut[T];
	fn next(&mut self) -> Option<Self::Item> {
		if self.size.y > 0 { Some(&mut self.take_mut(1).data[..self.size.x as usize]) }
		else { None }
	}
}

pub fn segment(total_length: u32, segment_count: u32) -> impl Iterator<Item=Range<u32>> {
	(0..segment_count)
	.map(move |i| (i+1)*total_length/segment_count)
	.scan(0, |start, end| { let next = (*start, end); *start = end; Some(next) })
	.map(|(start, end)| start..end)
}

impl<T> Image<&mut [T]> {
	#[track_caller] pub fn set<F:Fn(uint2)->T+Copy>(&mut self, f:F) { for y in 0..self.size.y { for x in 0..self.size.x { self[xy{x,y}] = f(xy{x,y}); } } }
	pub fn set_map<F:Fn(uint2,&T)->T+Copy>(&mut self, f:F) { for y in 0..self.size.y { for x in 0..self.size.x { self[xy{x,y}] = f(xy{x,y},&self[xy{x,y}]); } } }
	pub fn map<F:Fn(&T)->T+Copy>(&mut self, f:F) { for y in 0..self.size.y { for x in 0..self.size.x { self[xy{x,y}] = f(&self[xy{x,y}]); } } }
	pub fn zip_map<U, D:Deref<Target=[U]>, F: Fn(&T,&U)->T+Copy>(&mut self, source: &Image<D>, f: F) {
		assert!(self.size == source.size);
		for y in 0..self.size.y { for x in 0..self.size.x { self[xy{x,y}] = f(&self[xy{x,y}], &source[xy{x,y}]); } }
	}
	pub fn each_mut_zip_map<U, D:Deref<Target=[U]>, F: Fn(&mut T,&U)>(&mut self, source: &Image<D>, f: F) {
		assert!(self.size == source.size);
		for y in 0..self.size.y { for x in 0..self.size.x { f(&mut self[xy{x,y}], &source[xy{x,y}]); } }
	}
}

pub fn fill<T:Copy+Send>(target: &mut Image<&mut [T]>, value: T) { target.set(|_| value) }

impl<T> Image<Box<[T]>> {
	pub fn from_iter<I:IntoIterator<Item=T>>(size : size, iter : I) -> Self { Self::new(size, iter.into_iter().take((size.y*size.x) as usize).collect()) }
	pub fn from_xy<F:Fn(uint2)->T>(size : size, ref f: F) -> Self { Self::from_iter(size, (0..size.y).map(|y| (0..size.x).map(move |x| f(xy{x,y}))).flatten()) }
	#[cfg(feature="new_uninit")] pub fn uninitialized(size: size) -> Self { Self::new(size, unsafe{Box::new_uninit_slice((size.x * size.y) as usize).assume_init()}) }
}

impl<T:Copy> Image<Box<[T]>> {
	pub fn fill(size: size, value: T) -> Self { Self::from_iter(size, std::iter::from_fn(|| Some(value))) }
}
impl<D> Image<D> {
	//pub fn clone<T>(&self) -> Image<Box<[T]>> where T:Copy, D:AsRef<[T]> { Image::from_iter(self.size, self.as_ref().data.iter().copied()) }
	#[cfg(feature="slice_take")] pub fn clone<T>(&self) -> Image<Box<[T]>> where T:Copy, D:AsRef<[T]> { Image::from_iter(self.size, self.as_ref().map(|row| row).flatten().copied()) }
}
pub use num::Zero;
impl<T:Zero> Image<Box<[T]>> {
	pub fn zero(size: size) -> Self { Self::from_iter(size, std::iter::from_fn(|| Some(num::zero()))) }
}

impl<'t, T: bytemuck::Pod> Image<&'t [T]> {
	pub fn cast_slice<U:bytemuck::Pod>(slice: &'t [U], size: size) -> Self { Self::new(size, bytemuck::cast_slice(slice)) }
}

impl<'t, T: bytemuck::Pod> Image<&'t mut [T]> {
	#[track_caller] pub fn cast_slice_mut<U:bytemuck::Pod>(slice: &'t mut [U], size: size, stride: u32) -> Self { Self::strided(bytemuck::cast_slice_mut(slice), size, stride) }
}

use std::fmt::{Debug, Formatter, Result};
impl<D> Debug for Image<D> { fn fmt(&self, f: &mut Formatter) -> Result { assert!(self.stride == self.size.x); write!(f, "{:?}", self.size) } }

mod vector_bgr { vector::vector!(3 bgr T T T, b g r, Blue Green Red); } pub use vector_bgr::bgr;
mod vector_rgb { vector::vector!(3 rgb T T T, r g b, Red Green Blue); } pub use vector_rgb::rgb;
mod vector_bgra { vector::vector!(4 bgra T T T T, b g r a, Blue Green Red Alpha); } pub use vector_bgra::bgra;
mod vector_rgba { vector::vector!(4 rgba T T T T, r g b a, Red Blue Green Alpha); } pub use vector_rgba::rgba;
impl<T> rgba<T> { pub fn rgb(self) -> rgb<T> { let rgba{r,g,b,a:_} = self; rgb{r,g,b} } }
impl<T> From<bgra<T>> for rgba<T> { fn from(bgra{b,g,r,a}: bgra<T>) -> Self { rgba{r,g,b,a} } }

impl bgr<f32> { pub fn clamp(&self) -> Self { Self{b: self.b.clamp(0.,1.), g: self.g.clamp(0.,1.), r: self.r.clamp(0.,1.)} } }
pub type bgrf = bgr<f32>;
pub type rgbf = rgb<f32>;
pub type rgbaf = rgba<f32>;

impl From<u32> for bgr<u8> { fn from(bgr: u32) -> Self { bgr{b: (bgr >> 00) as u8 & 0xFF, g: (bgr >> 8) as u8 & 0xFF, r: (bgr >> 16) as u8 & 0xFF} } }
impl From<u32> for bgra<u8> { fn from(bgr: u32) -> Self { bgra{b: (bgr >> 00) as u8 & 0xFF, g: (bgr >> 8) as u8 & 0xFF, r: (bgr >> 16) as u8 & 0xFF, a: (bgr >> 24) as u8 & 0xFF} } }
impl From<bgr<u8>> for u32 { fn from(bgr{b,g,r}: bgr<u8>) -> Self { (0xFF << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32) } }

pub type bgr8 = bgr<u8>;
pub type rgb8 = rgb<u8>;
pub type bgra8 = bgra<u8>;
pub type rgba8 = rgba<u8>;

fn sRGB_OETF(linear: f64) -> f64 { if linear > 0.0031308 {1.055*linear.powf(1./2.4)-0.055} else {12.92*linear} }
use std::{array, sync::LazyLock};
pub static sRGB8_OETF12: LazyLock<[u8; 0x1000]> = LazyLock::new(|| array::from_fn(|i|(0xFF as f64 * sRGB_OETF(i as f64 / 0xFFF as f64)).round() as u8));
pub fn oetf8_12(oetf: &[u8; 0x1000], v: f32) -> u8 { oetf[(0xFFF as f32*v) as usize] } // 4K (fixme: interpolation of a smaller table might be faster)
pub fn sRGB8(v: f32) -> u8 { oetf8_12(&sRGB8_OETF12, v) } // FIXME: LazyLock::deref(sRGB_forward12) is too slow for image conversion
impl From<bgrf> for bgr8 { fn from(bgr: bgrf) -> Self { bgr.map(|c| sRGB8(c)) } }
impl From<bgrf> for u32 { fn from(bgr: bgrf) -> Self { bgr8::from(bgr).into() } }
pub fn sRGB8_from_linear(linear : &Image<&[f32]>) -> Image<Box<[u8]>> { let oetf = &sRGB8_OETF12; Image::from_iter(linear.size, linear.data.iter().map(|&v| oetf8_12(oetf, v))) }

pub const sRGB8_EOTF : LazyLock<[f32; 256]> = LazyLock::new(|| array::from_fn(|i| { let x = i as f64 / 255.; (if x > 0.04045 { ((x+0.055)/1.055).powf(2.4) } else { x / 12.92 }) as f32}));
pub fn eotf8(eotf: &[f32; 256], bgr: bgr8) -> bgrf { bgr.map(|c:u8| eotf[c as usize]) }
pub fn eotf8_rgb(eotf: &[f32; 256], bgr: rgb8) -> rgbf { bgr.map(|c:u8| eotf[c as usize]) }
//impl From<u32> for bgrf { fn from(bgr: u32) -> Self { eotf8(sRGB8_EOTF, bgr) } } // FIXME: LazyLock::deref(sRGB8_EOTF) is too slow for image conversion

//use num::Lerp; pub fn lerp(t: f32, a: u32, b: bgrf) -> u32 { u32::/*sRGB*/from(t.lerp(bgrf::/*sRGB⁻¹*/from(a), b)) }
pub fn oetf8_12_rgb(oetf: &[u8; 0x1000], bgr: bgrf) -> bgr<u8> { bgr.map(|c| oetf8_12(oetf, c)) } // 4K (fixme: interpolation of a smaller table might be faster)
pub fn lerp(eotf: &[f32; 256], oetf: &[u8; 0x1000], t: f32, a: bgr8, b: bgrf) -> u32 { oetf8_12_rgb(oetf, num::Lerp::lerp(&t, eotf8(eotf, a), b)).into() }
pub fn blend(mask : &Image<&[f32]>, target: &mut Image<&mut [u32]>, color: bgrf) {
	let (eotf, oetf) = (&sRGB8_EOTF, &sRGB8_OETF12);
	target.zip_map(mask, |&target, &t| lerp(eotf, oetf, t, bgr8::from(target), color));
}

/*impl From<u32> for bgr<u16> { fn from(bgr: u32) -> Self { bgr{b: (bgr >> 00) as u16 & 0x3FF, g: (bgr >> 10) as u16 & 0x3FF, r: (bgr >> 20) as u16 & 0x3FF} } }
impl From<bgr<u16>> for u32 { fn from(bgr{b,g,r}: bgr<u16>) -> Self { assert!(r<0x400 && g<0x400 && b<0x400,"{r} {g} {b}"); ((r as u32) << 20) | ((g as u32) << 10) | (b as u32) } }

pub fn invert(image: &mut Image<&mut [u32]>, m: bgr<bool>) { image.map(|&bgr| { m.zip(bgr::<u16>::from(bgr)).map(|(m,c)| if m {0x3FF-c} else {c}).into()}) }

// PQ
const m1 : f64 = 2610./16384.;
const m2 : f64 = 128.*2523./4096.;
const c2 : f64 = 32.*2413./4096.;
const c3 : f64 = 32.*2392./4096.;
const c1 : f64 = c3-c2+1.;

pub fn PQ_EOTF(pq: f64) -> f64 { (f64::max(0., pq.powf(1./m2)-c1) / (c2 - c3*pq.powf(1./m2))).powf(1./m1) } // /10Kcd
use std::{array, sync::LazyLock};
pub static PQ10_EOTF : LazyLock<[f32; 0x400]> = LazyLock::new(|| std::array::from_fn(|pq|PQ_EOTF(pq as f64 / 0x3FF as f64) as f32));
pub fn PQ10(x: f32) -> u16 {
	let (mut left, mut right) = (0, PQ10_EOTF.len()); while left < right { let mid = left + (right - left) / 2; if PQ10_EOTF[mid] < x { left = mid + 1; } else { right = mid; } }
	left as u16
}
impl From<bgrf> for u32 { fn from(bgr: bgrf) -> Self { bgr.map(|c| PQ10(c)).into() } }
pub fn PQ10_from_linear(linear : &Image<&[f32]>) -> Image<Box<[u16]>> { Image::from_iter(linear.size, linear.data.iter().map(|&v| PQ10(v))) }

pub fn from_PQ10(pq: u16) -> f32 { PQ10_EOTF[pq as usize] } // FIXME: LazyLock::deref(PQ10_EOTF) is too slow for image conversion
impl From<u32> for bgrf { fn from(bgr: u32) -> Self { bgr::from(bgr).map(|c| from_PQ10(c)) } }

pub fn lerp(t: f32, a: u32, b: bgrf) -> u32 { u32::/*PQ10*/from(t.lerp(bgrf::/*PQ10⁻¹*/from(a), b)) }
pub fn blend(mask : &Image<&[f32]>, target: &mut Image<&mut [u32]>, color: bgrf) { target.zip_map(mask, |&target, &t| lerp(t, target, color)); }

pub fn PQ_OETF(y: f64) -> f64 { ((c1+c2*y.powf(m1))/(1.+c3*y.powf(m1))).powf(m2) }
pub fn PQ10_OETF(v: f64) -> u16 {
	assert!(v >= 0. && v <= 1.);
	let PQ = PQ_OETF(v);
	assert!(PQ <= 1.);
	(1023.*PQ).round() as u16
}

pub const sRGB_to_PQ10 : LazyLock<[u16; 256]> = LazyLock::new(|| array::from_fn(|i| { let x = i as f64 / 255.;
    PQ10_OETF(if x > 0.04045 { ((x+0.055)/1.055).powf(2.4) } else { x / 12.92 })
}));
pub fn sRGB8_to_PQ10(eetf: &[u16; 256], rgb{r,g,b}: rgb8) -> u32 { bgr{b: eetf[b as usize], g: eetf[g as usize], r: eetf[r as usize]}.into() }
//impl From<rgb8> for u32 { fn from(rgb: rgb8) -> Self { rgb8_to_PQ10(sRGB_to_PQ10, rgb) } } // FIXME: LazyLock::deref(sRGB_to_PQ10) is too slow for image conversion*/

#[cfg(feature="io")]
pub fn u8(path: impl AsRef<std::path::Path>) -> Image<Box<[u8]>> {
	let image = image::open(path).unwrap().into_luma8();
	assert_eq!(image.sample_layout(), image::flat::SampleLayout{channels: 1, channel_stride: 1, width: image.width(), width_stride: 1, height: image.height(), height_stride: image.width() as _});
	Image::new(xy{x: image.width(), y: image.height()}, image.into_raw().into_boxed_slice())
}

#[cfg(feature="io")]
pub fn rgb8(path: impl AsRef<std::path::Path>) -> Image<Box<[rgb8]>> {
	let image = image::open(path).unwrap().into_rgb8();
	assert_eq!(image.sample_layout(), image::flat::SampleLayout{channels: 3, channel_stride: 1, width: image.width(), width_stride: 3, height: image.height(), height_stride: 3*image.width() as usize});
	Image::new(xy{x: image.width(), y: image.height()}, bytemuck::cast_slice_box(image.into_raw().into_boxed_slice()))
}

#[cfg(feature="io")]
pub fn rgba8(path: impl AsRef<std::path::Path>) -> Image<Box<[rgba8]>> {
	let image = image::open(path).unwrap().into_rgba8();
	assert_eq!(image.sample_layout(), image::flat::SampleLayout{channels: 4, channel_stride: 1, width: image.width(), width_stride: 4, height: image.height(), height_stride: 4*image.width() as usize});
	Image::new(xy{x: image.width(), y: image.height()}, bytemuck::cast_slice_box(image.into_raw().into_boxed_slice()))
}

#[cfg(feature="png")]
pub fn u16(path: impl AsRef<std::path::Path>) -> Image<Box<[u16]>> {
	let file = std::fs::File::open(path).unwrap();
	let decoder = png::Decoder::new(file);
	let mut reader = decoder.read_info().unwrap();
	let mut buffer = vec![0; reader.output_buffer_size()];
	reader.next_frame(&mut buffer).unwrap();
	let mut buffer_u16 = vec![0; (reader.info().width * reader.info().height) as usize];
	let mut buffer_cursor = std::io::Cursor::new(buffer);
	use byteorder::ReadBytesExt;
	buffer_cursor.read_u16_into::<byteorder::BigEndian>(&mut buffer_u16).unwrap();
	Image::new(xy{x: reader.info().width, y: reader.info().height}, buffer_u16.into_boxed_slice())
}
