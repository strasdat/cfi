// build.rs

fn main() {

    cxx_build::bridge("src/main.rs")
        .file("src/ceres_wrapper.cc")
        .flag("-I/usr/include/eigen3/")
        .compile("metal-carl");
    println!("cargo:rustc-link-lib=ceres");

}
