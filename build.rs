#[cfg(feature = "loader_prophesee")]
use std::path::PathBuf;
#[cfg(feature = "loader_prophesee")]
use cxx_build::CFG;
#[cfg(feature = "loader_prophesee")]
use pkg_config::probe_library;

fn main() {
  #[cfg(not(any(feature = "loader_render", feature = "loader_prophesee")))]
  panic!("At least one of `render` and `prophesee` features must be enabled!");
  #[cfg(feature = "loader_prophesee")]
  {
    let opencv4 = probe_library("opencv4").unwrap();
    let opencv4_include_path = opencv4.include_paths.iter().map(PathBuf::as_path);
    CFG.exported_header_dirs.extend(opencv4_include_path);
    cxx_build::bridge("src/loader/loader_prophesee.rs")
      .file("src/loader/loader_prophesee.cc")
      .flag_if_supported("-std=c++14")
      .compile("event_ps");
    println!("cargo:rerun-if-changed=src/loader/loader_prophesee.rs");
    println!("cargo:rerun-if-changed=src/loader/loader_prophesee.cc");
    println!("cargo:rerun-if-changed=include/loader_prophesee.h");
    println!("cargo:rustc-link-lib=metavision_hal");
    println!("cargo:rustc-link-lib=metavision_sdk_driver");
  }
}
