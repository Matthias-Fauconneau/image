#![cfg_attr(feature="type_alias_impl_trait",feature(type_alias_impl_trait))]
#![cfg_attr(feature="slice_take",feature(slice_take))]
#![cfg_attr(feature="const_trait_impl",feature(const_trait_impl))]

#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]
pub use vector::{xy, uint2, size, int2};
use vector::Rect;

#[derive(std::hash::Hash)]
pub struct Image<D> {
	pub data : D,
	pub size : size,
	pub stride : u32,
}

impl<D> Image<D> {
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
	pub unsafe fn get_unchecked(&self, x: u32, y: u32) -> &T { unsafe { self.data.get_unchecked((y * self.stride + x) as usize) } }
	pub fn rows(&self, rows: Range<u32>) -> Image<&[T]> {
		assert!(rows.end <= self.size.y);
		Image{size: xy{x: self.size.x, y: rows.len() as u32}, stride: self.stride, data: &self.data[(rows.start*self.stride) as usize..]}
	}
	#[track_caller] pub fn slice(&self, offset: uint2, size: size) -> Image<&[T]> {
		assert!(offset.x+size.x <= self.size.x && offset.y+size.y <= self.size.y, "{:?} {:?} {:?} {:?}", offset, size, self.size, offset+size);
		let start = offset.y*self.stride+offset.x;
		Image{size, stride: self.stride, data: &self.data[start as usize..(start+(size.y-1)*self.stride+size.x) as usize]}
	}
}

impl<T, D:DerefMut<Target=[T]>> Image<D> {
	pub fn get_mut(&mut self, i: uint2) -> Option<&mut T> { self.index(i).map(|i| &mut self[i]) }
	pub unsafe fn get_unchecked_mut(&mut self, x: u32, y: u32) -> &mut T { unsafe { self.data.get_unchecked_mut((y * self.stride + x) as usize) } }
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
		Image{size: xy{x: self.size.x, y: mid}, stride: self.stride, data: self.data.take(..((mid-1)*self.stride + if self.size.y == 0 { self.size.x } else { self.stride }) as usize).unwrap()}
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
	#[track_caller] fn next(&mut self) -> Option<Self::Item> {
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

impl<T> Image<&mut [T]> {
	#[track_caller] pub fn set<F:Fn(uint2)->T+Copy>(&mut self, f:F) { for y in 0..self.size.y { for x in 0..self.size.x { self[xy{x,y}] = f(xy{x,y}); } } }
	pub fn zip_map<U, D:Deref<Target=[U]>, F: Fn(&T,&U)->T+Copy>(&mut self, source: &Image<D>, f: F) {
		assert!(self.size == source.size);
		for y in 0..self.size.y { for x in 0..self.size.x { self[xy{x,y}] = f(&self[xy{x,y}], &source[xy{x,y}]); } }
	}
}

pub fn fill<T:Copy+Send>(target: &mut Image<&mut [T]>, value: T) { target.set(|_| value) }

impl<T> Image<Box<[T]>> {
	#[track_caller] pub fn from_iter<I:IntoIterator<Item=T>>(size : size, iter : I) -> Self { Self::new(size, iter.into_iter().take((size.y*size.x) as usize).collect()) }
	pub fn from_xy<F:Fn(uint2)->T>(size : size, ref f: F) -> Self { Self::from_iter(size, (0..size.y).map(|y| (0..size.x).map(move |x| f(xy{x,y}))).flatten()) }
	pub fn uninitialized(size: size) -> Self { Self::new(size, unsafe{Box::new_uninit_slice((size.x * size.y) as usize).assume_init()}) }
}

impl<T:Copy> Image<Box<[T]>> {
	pub fn fill(size: size, value: T) -> Self { Self::from_iter(size, std::iter::from_fn(|| Some(value))) }
}

impl<D: IntoIterator> Image<D> { pub fn map<T>(self, f: impl FnMut(<D as IntoIterator>::Item)->T) -> Image<Box<[T]>> { Image::from_iter(self.size, self.data.into_iter().map(f)) } }

impl<D> Image<D> {
	pub fn map_xy<T, F:Fn(uint2,&<Self as Index<uint2>>::Output)->T>(&self, f: F) -> Image<Box<[T]>> where Self: Index<uint2> {
		Image::from_xy(self.size, |p| f(p,&self[p]))
	}
}

impl<D> Image<D> {
	#[cfg(feature="slice_take")] pub fn clone<T>(&self) -> Image<Box<[T]>> where D: AsRef<[T]>, T:Clone {
		Image::from_iter(self.size, Iterator::map(self.as_ref(), |row| row).flatten().cloned())
	}
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

use std::fmt::{self, Debug, Formatter};
impl<D> Debug for Image<D> { fn fmt(&self, f: &mut Formatter) -> fmt::Result { assert!(self.stride == self.size.x); write!(f, "{:?}", self.size) } }

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

pub type bgr8 = bgr<u8>;
pub type rgb8 = rgb<u8>;
pub type bgra8 = bgra<u8>;
pub type rgba8 = rgba<u8>;

pub fn rgba8_from_u8(image: &Image<impl AsRef<[u8]>>) -> Image<Box<[rgba8]>> { image.as_ref().map(|&v| rgba::from(rgba{r:v,g:v,b:v,a:0xFF})) }
pub fn rgba8_from_rgb8(image: &Image<impl AsRef<[rgb8]>>) -> Image<Box<[rgba8]>> { image.as_ref().map(|&rgb{r,g,b}| rgba::from(rgba{r,g,b,a:0xFF})) }

impl From<u32> for bgr8 { fn from(bgr: u32) -> Self { bgr{b: (bgr>>0) as u8 & 0xFF, g: (bgr>>8) as u8 & 0xFF, r: (bgr>>16) as u8 & 0xFF} } }
impl From<u32> for bgra8 { fn from(bgr: u32) -> Self { bgra{b: (bgr>>0) as u8 & 0xFF, g: (bgr>>8) as u8 & 0xFF, r: (bgr>>16) as u8 & 0xFF, a: (bgr>>24) as u8 & 0xFF} } }
impl From<bgr8> for u32 { fn from(bgr{b,g,r}: bgr<u8>) -> Self { (0xFF << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32) } }

pub fn lerp_rgb8(t: f32, a: rgb8, b: rgb8) -> rgb8 { let t8:u16=(t*(0x100 as f32)) as u16; rgb::<u8>::from(((0x100-t8)*rgb::<u16>::from(a) + t8*rgb::<u16>::from(b))/0x100) }

fn sRGB_OETF(linear: f64) -> f64 { if linear > 0.0031308 {1.055*linear.powf(1./2.4)-0.055} else {12.92*linear} }
use std::{array, sync::LazyLock};
pub static sRGB8_OETF12: LazyLock<[u8; 0x1000]> = LazyLock::new(|| array::from_fn(|i|(0xFF as f64 * sRGB_OETF(i as f64 / 0xFFF as f64)).round() as u8));
pub fn oetf8_12(oetf: &[u8; 0x1000], v: f32) -> u8 { oetf[(0xFFF as f32*v) as usize] } // 4K (fixme: interpolation of a smaller table might be faster)
pub fn sRGB8(v: f32) -> u8 { oetf8_12(&sRGB8_OETF12, v) } // FIXME: LazyLock::deref(sRGB_forward12) is too slow for image conversion
pub fn oetf8_12_bgr(oetf: &[u8; 0x1000], bgr: bgrf) -> bgr<u8> { bgr.map(|c| oetf8_12(oetf, c)) }
pub fn oetf8_12_rgb(oetf: &[u8; 0x1000], rgb: rgbf) -> rgb<u8> { rgb.map(|c| oetf8_12(oetf, c)) }
pub fn bgr8_from(bgr: bgrf) -> bgr8 { bgr.map(|c| sRGB8(c)) }
impl From<bgrf> for u32 { fn from(bgr: bgrf) -> Self { u32::from(bgr8_from(bgr)) } }
pub fn sRGB8_from_linear(linear : &Image<&[f32]>) -> Image<Box<[u8]>> { let oetf = &sRGB8_OETF12; Image::from_iter(linear.size, linear.data.iter().map(|&v| oetf8_12(oetf, v))) }

pub const sRGB8_EOTF : LazyLock<[f32; 256]> = LazyLock::new(|| array::from_fn(|i| { let x = i as f64 / 255.; (if x > 0.04045 { ((x+0.055)/1.055).powf(2.4) } else { x / 12.92 }) as f32}));
pub fn eotf8(eotf: &[f32; 256], bgr: bgr8) -> bgrf { bgr.map(|c:u8| eotf[c as usize]) }
pub fn eotf8_rgb(eotf: &[f32; 256], bgr: rgb8) -> rgbf { bgr.map(|c:u8| eotf[c as usize]) }
//impl From<u32> for bgrf { fn from(bgr: u32) -> Self { eotf8(sRGB8_EOTF, bgr) } } // FIXME: LazyLock::deref(sRGB8_EOTF) is too slow for image conversion
pub fn eotf(u8: &Image<impl AsRef<[rgb8]>>) -> Image<Box<[rgbf]>> { let eotf = &sRGB8_EOTF; Image::from_iter(u8.size, u8.data.as_ref().iter().map(|&v| eotf8_rgb(eotf, v))) }

//use num::Lerp; pub fn lerp(t: f32, a: u32, b: bgrf) -> u32 { u32::/*sRGB*/from(t.lerp(bgrf::/*sRGB⁻¹*/from(a), b)) }
pub fn lerp(eotf: &[f32; 256], oetf: &[u8; 0x1000], t: f32, a: bgr8, b: bgrf) -> u32 { oetf8_12_bgr(oetf, num::lerp(t, eotf8(eotf, a), b)).into() }
pub fn blend(mask : &Image<&[f32]>, target: &mut Image<&mut [u32]>, color: bgrf) {
	let (eotf, oetf) = (&sRGB8_EOTF, &sRGB8_OETF12);
	target.zip_map(mask, |&target, &t| lerp(eotf, oetf, t, bgr8::from(target), color));
}

// /!\ keeps floats in the initial 8bit space (sRGB), i.e incorrect for interpolations
pub fn from_u8(image: &Image<impl AsRef<[u8]>>) -> Image<Box<[f32]>> { image.as_ref().map(|&u8| f32::from(u8)) }
pub fn from_rgb8(image: &Image<impl AsRef<[rgb8]>>) -> Image<Box<[rgbf]>> { image.as_ref().map(|&rgb8| rgbf::from(rgb8)) }
pub fn from_rgbf(image: &Image<impl AsRef<[rgbf]>>) -> Image<Box<[rgb8]>> { image.as_ref().map(|&rgbf| rgb8::from(rgbf)) }
pub fn from_rgbaf(image: &Image<impl AsRef<[rgbaf]>>) -> Image<Box<[rgba8]>> { image.as_ref().map(|&rgbaf| rgba8::from(rgbaf)) }

pub fn nearest<T:Copy>(size: size, source: &Image<impl Deref<Target=[T]>>) -> Image<Box<[T]>> { Image::from_xy(size, |xy{x,y}| source[xy{x,y}*source.size/size]) }

pub fn downsample_sum<const FACTOR: u32, T:Copy+std::iter::Sum>(source: &Image<impl std::ops::Deref<Target=[T]>>) -> Image<Box<[T]>> {
	Image::from_xy(source.size/FACTOR, |xy{x,y}| (0..FACTOR).map(|dy| (0..FACTOR).map(move |dx| source[xy{x:x*FACTOR+dx,y:y*FACTOR+dy}])).flatten().sum::<T>())
}

pub fn downsample<T: Copy, D: Deref<Target=[T]>, F, const FACTOR: u32>(ref source: Image<D>) -> Image<Box<[F::Output]>>
	where F: From<T>+std::iter::Sum<F>+std::ops::Div<f32> {
	Image::from_xy(source.size/FACTOR, |xy{x,y}|
			(0..FACTOR).map(|dy| (0..FACTOR).map(move |dx| F::from(source[xy{x:x*FACTOR+dx,y:y*FACTOR+dy}])))
				.flatten().sum::<F>() / (FACTOR as f32*FACTOR as f32)
	)
}

pub fn downsample_rgba<const FACTOR: u32>(ref source: Image<&[rgba<f32>]>) -> Image<Box<[rgba<f32>]>> {
	Image::from_xy(source.size/FACTOR, |xy{x,y}| {
		let rgba{r,g,b,a} = (0..FACTOR).map(|dy| (0..FACTOR).map(move |dx| { let rgba{r,g,b, a} = source[xy{x:x*FACTOR+dx,y:y*FACTOR+dy}]; rgba{r: r*a, g: g*a, b: b*a, a} })).flatten().sum::<rgba<f32>>();
		if a == 0. { return rgba{r: 0., g: 0., b: 0., a} }
		rgba{r: r / a, g: g / a, b: b /a, a: a / (FACTOR*FACTOR) as f32}
	})
}

pub use vector::vec2;
use num::Lerp;
pub fn bilinear_sample<D>(image@Image{size,..}: &Image<D>, s: vec2) -> <Image<D> as Index<uint2>>::Output where Image<D>: Index<uint2>, <Image<D> as Index<uint2>>::Output: Sized+Copy, f32: Lerp<<Image<D> as Index<uint2>>::Output> {
	let i = uint2::from(s);
	let [n00, n10, n01, n11] = [xy{x: 0, y: 0},xy{x: 1, y: 0},xy{x: 0, y: 1},xy{x: 1, y:1}].map(|d| image[xy{x: ((i.x as i32+d.x) as u32+size.x)%size.x, y: ((i.y as i32+d.y).max(0) as u32).min(size.y-1)}]);
	let f = s-s.map(f32::floor);
	use num::lerp;
	lerp(f.y, lerp(f.x, n00, n10), lerp(f.x, n01, n11))
}

#[cfg(feature="io")]
pub fn bilinear<D>(target_size: size, image@Image{size,..}: &Image<D>) -> Image<Box<[f32]>> where Image<D>: Index<uint2>, <Image<D> as Index<uint2>>::Output: Copy+Into<f32> {
	let ref image = image::imageops::resize(&image::ImageBuffer::from_fn(size.x, size.y, |x, y| image::Luma([image[xy{x,y}].into()])), target_size.x, target_size.y, image::imageops::FilterType::Triangle);
	Image::from_iter(target_size, (0..target_size.y).map(|y| (0..target_size.x).map(move |x| image.get_pixel(x,y).0[0])).flatten())
}

/*pub fn bilinear<D>(target_size: size, image@Image{size,..}: &Image<D>) -> Image<Box<[<Image<D> as Index<uint2>>::Output]>> where Image<D>: Index<uint2>, <Image<D> as Index<uint2>>::Output: Sized+Copy, f32: Lerp<<Image<D> as Index<uint2>>::Output> {
	let scale = xy::<f32>::from(*size)/xy::<f32>::from(target_size);
	Image::from_iter(target_size, (0..target_size.y).map(|y| (0..target_size.x).map(move |x| bilinear_sample(image, scale*xy{x: x as f32,y: y as f32}))).flatten())
}*/

#[cfg(feature="io")]
pub fn bilinear_rgb8<D>(target_size: size, image@Image{size,..}: &Image<D>) -> Image<Box<[rgb8]>> where Image<D>: Index<uint2>, <Image<D> as Index<uint2>>::Output: Copy+Into<[u8; 3]> {
	let ref image = image::imageops::resize(&image::ImageBuffer::from_fn(size.x, size.y, |x, y| image::Rgb(image[xy{x,y}].into())), target_size.x, target_size.y, image::imageops::FilterType::Triangle);
	Image::from_iter(target_size, (0..target_size.y).map(|y| (0..target_size.x).map(move |x| image.get_pixel(x,y).0.into())).flatten())
}

pub fn transpose_box_convolve<const R: u32>(source@Image{size,..}: Image<&[f32]>) -> Image<Box<[f32]>> {
	let mut transpose = Image::uninitialized(size.yx());
	for y in 0..size.y {
		let r : f64 = R as f64;
		let mut sum : f64 = source[xy{x: 0, y}] as f64*r;
		for x in 0..R { sum += source[xy{x, y}] as f64; }
		for x in 0..R {
			sum += source[xy{x: x+R, y}] as f64;
			transpose[xy{x: y, y: x}] = (sum/(r+1.+r)) as f32;
			sum -= source[xy{x: 0, y}] as f64;
		}
		for x in R..size.x-R {
			sum += source[xy{x: x+R, y}] as f64;
			transpose[xy{x: y, y: x}] = (sum/(r+1.+r)) as f32;
			sum -= source[xy{x: (x as i32-R as i32) as u32, y}] as f64;
		}
		for x in size.x-R..size.x {
				sum += source[xy{x: size.x-1, y}] as f64;
				transpose[xy{x: y, y: x}] = (sum/(r+1.+r)) as f32;
				sum -= source[xy{x: (x as i32-R as i32) as u32, y}] as f64;
		}
	}
	transpose
}

pub fn blur_xy<const X: u32, const Y: u32>(image: &Image<impl AsRef<[f32]>>) -> Image<Box<[f32]>> {
	transpose_box_convolve::<Y>(transpose_box_convolve::<X>(image.as_ref()).as_ref())
}

pub fn blur<const R: u32>(image: &Image<impl AsRef<[f32]>>) -> Image<Box<[f32]>> { blur_xy::<R,R>(image) }

pub fn transpose_box_convolve_rgb<const R: u32>(source: Image<&[rgb<f32>]>) -> Image<Box<[rgb<f32>]>> {
	let mut transpose = Image::uninitialized(source.size.yx());
	for y in 0..source.size.y {
		let r : f64 = R as f64;
		let mut sum = r*rgb::<f64>::from(source[xy{x: 0, y}]);
		for x in 0..R { sum += rgb::<f64>::from(source[xy{x, y}]); }
		for x in 0..R {
			sum += rgb::<f64>::from(source[xy{x: x+R, y}]);
			transpose[xy{x: y, y: x}] = rgb::<f32>::from(sum/(r+1.+r));
			sum -= rgb::<f64>::from(source[xy{x: 0, y}]);
		}
		for x in R..source.size.x-R {
			sum += rgb::<f64>::from(source[xy{x: x+R, y}]);
			transpose[xy{x: y, y: x}] = rgb::<f32>::from(sum/(r+1.+r));
			sum -= rgb::<f64>::from(source[xy{x: (x as i32-R as i32) as u32, y}]);
		}
		for x in source.size.x-R..source.size.x {
				sum += rgb::<f64>::from(source[xy{x: source.size.x-1, y}]);
				transpose[xy{x: y, y: x}] = rgb::<f32>::from(sum/(r+1.+r));
				sum -= rgb::<f64>::from(source[xy{x: (x as i32-R as i32) as u32, y}]);
		}
	}
	transpose
}

pub fn blur_rgb<const R: u32>(image: &Image<impl AsRef<[rgb<f32>]>>) -> Image<Box<[rgb<f32>]>> {
	transpose_box_convolve_rgb::<R>(transpose_box_convolve_rgb::<R>(image.as_ref()).as_ref())
}

pub fn blur_rgb8<const R: u32>(image: &Image<impl AsRef<[rgb8]>>) -> Image<Box<[rgb8]>> { // FIXME: not sRGB
	from_rgbf(&blur_rgb::<R>(&from_rgb8(image)))
}

pub fn transpose_box_convolve_rgba<const R: u32>(source: Image<&[rgba<f32>]>) -> Image<Box<[rgba<f32>]>> {
	let mut transpose = Image::uninitialized(source.size.yx());
	for y in 0..source.size.y {
		let r : f64 = R as f64;
		let mut sum = r*rgba::<f64>::from(source[xy{x: 0, y}]);
		for x in 0..R { sum += rgba::<f64>::from(source[xy{x, y}]); }
		for x in 0..R {
			sum += rgba::<f64>::from(source[xy{x: x+R, y}]);
			transpose[xy{x: y, y: x}] = rgba::<f32>::from(sum/(r+1.+r));
			sum -= rgba::<f64>::from(source[xy{x: 0, y}]);
		}
		for x in R..source.size.x-R {
			sum += rgba::<f64>::from(source[xy{x: x+R, y}]);
			transpose[xy{x: y, y: x}] = rgba::<f32>::from(sum/(r+1.+r));
			sum -= rgba::<f64>::from(source[xy{x: (x as i32-R as i32) as u32, y}]);
		}
		for x in source.size.x-R..source.size.x {
				sum += rgba::<f64>::from(source[xy{x: source.size.x-1, y}]);
				transpose[xy{x: y, y: x}] = rgba::<f32>::from(sum/(r+1.+r));
				sum -= rgba::<f64>::from(source[xy{x: (x as i32-R as i32) as u32, y}]);
		}
	}
	transpose
}

pub fn blur_rgba<const R: u32>(image: &Image<impl AsRef<[rgba<f32>]>>) -> Image<Box<[rgba<f32>]>> {
	transpose_box_convolve_rgba::<R>(transpose_box_convolve_rgba::<R>(image.as_ref().map(|&rgba{r,g,b,a}| rgba{r: r*a, g: g*a, b: b*a, a}).as_ref()).as_ref()).map(|rgba{r,g,b,a}| rgba{r: r/a, g: g/a, b: b/a, a})
}

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

#[cfg(feature="io")] pub type Result<T=(),E=image::ImageError> = std::result::Result<T, E>;
#[cfg(feature="io")]
pub fn save_u8(path: impl AsRef<std::path::Path>, Image{size, data, stride}: &Image<impl Deref<Target=[u8]>>) -> Result {
	assert_eq!(*stride, size.x);
	image::save_buffer(path, &data, size.x, size.y, image::ColorType::L8)
}
#[cfg(feature="io")]
pub fn save_alpha(path: impl AsRef<std::path::Path>, Image{size, data, stride}: &Image<impl Deref<Target=[f32]>>) -> Result {
	assert_eq!(*stride, size.x);
	image::save_buffer(path, &Box::from_iter(data.into_iter().map(|&a| (a*(0xFF as f32)) as u8)), size.x, size.y, image::ColorType::L8)
}
#[cfg(feature="io")]
pub fn save_rgb(path: impl AsRef<std::path::Path>, Image{size, data, stride}: &Image<impl Deref<Target=[rgb8]>>) -> Result {
	assert_eq!(*stride, size.x);
	image::save_buffer(path, bytemuck::cast_slice(&data), size.x, size.y, image::ColorType::Rgb8)
}
#[cfg(feature="io")]
pub fn save_rgba(path: impl AsRef<std::path::Path>, Image{size, data, stride}: &Image<impl Deref<Target=[rgba8]>>) -> Result {
	assert_eq!(*stride, size.x);
	image::save_buffer(path, bytemuck::cast_slice(&data), size.x, size.y, image::ColorType::Rgba8)
}

#[cfg(feature="exr")] pub type ExrResult<T=(),E=exr::error::Error> = std::result::Result<T, E>;

#[cfg(feature="exr")]
pub fn f32(path: impl AsRef<std::path::Path>) -> ExrResult<Image<Box<[f32]>>> {
	let exr = exr::prelude::read_first_flat_layer_from_file(path)?.layer_data;
	let size = {let exr::prelude::Vec2(x,y) = exr.size; xy{x: x as u32,y: y as _}};
	Ok(Image::from_xy(size, |xy{x,y}| { let &[exr::prelude::Sample::F32(v)] = exr.sample_vec_at(exr::prelude::Vec2(x as _,y as _)).as_slice() else {unreachable!()}; v }))
}

#[cfg(feature="exr")]
pub fn save_exr<D: std::ops::Deref<Target=[f32]>+Sync>(path: impl AsRef<std::path::Path>, channel: &str, image@Image{size, ..}: &Image<D>) -> ExrResult {
	use exr::prelude::*;
	Image::from_channels(Vec2(size.x as _, size.y as _), SpecificChannels{
		channels: (ChannelDescription::named(channel, SampleType::F32),),
		pixels: |Vec2(x,y)| (image[xy{x: x as _,y: y as _}],)
	}).write().to_file(path)
}

vector::vector!(2 za T T, z a, Depth Opacity);

#[cfg(feature="exr")]
pub fn za(path: impl AsRef<std::path::Path>) -> Image<Box<[za<f32>]>> {
	let exr = exr::prelude::read_first_flat_layer_from_file(path).unwrap().layer_data;
	Image::from_xy({let exr::prelude::Vec2(x,y) = exr.size; xy{x: x as u32,y: y as _}}, |xy{x,y}| if let &[exr::prelude::Sample::F32(z), exr::prelude::Sample::F32(a)] = exr.sample_vec_at(exr::prelude::Vec2(x as _,y as _)).as_slice() { za{z,a} } else { unimplemented!("") })
}

#[cfg(feature="exr")]
pub fn save_exr2<D: std::ops::Deref<Target=[za<f32>]>+Sync>(path: impl AsRef<std::path::Path>, image@Image{size, ..}: &Image<D>) -> exr::error::Result<()> {
	use exr::prelude::*;
	Image::from_channels(Vec2(size.x as _, size.y as _), SpecificChannels{
		channels:(ChannelDescription::named("depth", SampleType::F32), ChannelDescription::named("opacity", SampleType::F32)),
		pixels:|Vec2(x,y)| { let za{z,a} = image[xy{x: x as _,y: y as _}]; (z,a) }
	}).write().to_file(path)
}

use vector::{minmax, MinMax};
pub fn bbox(mask: &Image<impl Deref<Target=[f32]>>) -> MinMax<uint2> {
	minmax((0..mask.size.y).map(|y| (0..mask.size.x).map(move |x| xy{x,y})).flatten().filter(|&p| mask[p] > 0.)).unwrap()
}

impl<T, D:Deref<Target=[T]>> Image<D> {
	#[track_caller] pub fn crop(&self, bbox: MinMax<int2>) -> Image<&[T]> {
		let bbox = bbox.clip(MinMax{min: xy{x:0,y:0}, max: self.size.signed()}).unsigned();
		self.slice(bbox.min, bbox.size())
	}
}
impl<T, D:DerefMut<Target=[T]>> Image<D> {
	#[track_caller] pub fn crop_mut(&mut self, bbox: MinMax<int2>) -> Image<&mut [T]> {
		let bbox = bbox.clip(MinMax{min: xy{x:0,y:0}, max: self.size.signed()}).unsigned();
		self.slice_mut(bbox.min, bbox.size())
	}
}

/*#[cfg(feature="median")]
pub fn median<D>(target_size: size, image@Image{size,..}: &Image<D>) -> Image<Box<[rgb8]>> where Image<D>: Index<uint2>, <Image<D> as Index<uint2>>::Output: Copy+Into<[u8; 3]> {
	let ref image = imageproc::filter::median_filter(&image::ImageBuffer::from_fn(size.x, size.y, |x, y| image::Rgb(image[xy{x,y}].into())), 256, 256);
	Image::from_iter(target_size, (0..target_size.y).map(|y| (0..target_size.x).map(move |x| image.get_pixel(x,y).0.into())).flatten())
}**/
