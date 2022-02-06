
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod generic_ga;
use generic_ga::*;

#[pyfunction]
fn run_generic_ga(iterations: u32, 
                 k : usize, length : usize,
                 mut_rate : f64, cross_rate : f64,
                 elitism : usize, minimizing : bool, init_rand : bool, 
                 pop_size : usize,
                 metrics_filename : String) -> PyResult<String> {
    generic_ga(iterations, k, length, mut_rate, cross_rate, elitism, minimizing, init_rand, pop_size, metrics_filename)
        .map_err(|err| println!("Generic GA Failed with: {:?}", err)).ok();
    Ok("Done!".to_string())
}


#[pymodule]
fn libcarbon(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_generic_ga, m)?)?;
    Ok(())
}
