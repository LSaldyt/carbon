
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod regev;
use regev::*;

mod simple_ga;
use simple_ga::*;

#[pyfunction]
fn run_simple_ga(a: u32) -> PyResult<String> {
    simple_ga(a);
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
