
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod regev;
use regev::*;

mod simple_ga;
use simple_ga::*;

#[pyfunction]
fn run_simple_ga(iterations: u32, 
                 k : usize, length : usize,
                 pop_size : usize,
                 metrics_filename : String) -> PyResult<String> {
    simple_ga(iterations, k, length, pop_size, metrics_filename)
        .map_err(|err| println!("Simple GA Failed with: {:?}", err)).ok();
    Ok("Done!".to_string())
}

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    regev(10000);
    Ok((a + b).to_string())
}

#[pymodule]
fn libcarbon(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(run_simple_ga, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
