use rand::Rng;
// use std::assert;
use std::collections::VecDeque;
use ordered_float::OrderedFloat;
type Of64 = OrderedFloat<f64>;

use statistical::*; // Hail satan

type Member = Vec<i32>; // In our problem, members are simple vectors
type Population = VecDeque<Member>; // Population is normal vec of members

struct Problem<'a> {
    rng: &'a mut rand::rngs::ThreadRng,
    min: i32,
    max: i32,
    length: usize,
    pop_size: usize,
    fitness: fn(&Member) -> Of64,
    minimizing: bool,
    max_fit : Of64,
    min_fit : Of64
}

#[derive(Debug)]
struct Metrics {
    min   : Of64,
    max   : Of64,
    avg   : Of64,
    stdev : Of64
}

fn num(min: i32, max: i32, rng : &mut rand::rngs::ThreadRng) -> i32 {
    // Silly convenience function because "gen_range"
    // is much longer to type than "num"
    return rng.gen_range(min..max+1);
}

fn generate(problem: &mut Problem) -> Member {
    // Generate a random length l vector constrained to a range (min, max)
    // The first "bit" is reserved for sign
    let mut initial = vec![0; problem.length];
    initial[0] = num(0, 1, problem.rng);
    for i in 1..problem.length {
        initial[i] = num(problem.min, problem.max, problem.rng);
    }
    return initial
}

fn initialize(problem: &mut Problem) -> Population {
    // Create the initial population
    let mut population: Population = VecDeque::with_capacity(problem.pop_size);
    for _i in 0..problem.pop_size {
        population.push_back(generate(problem));
    }
    return population
}

fn mutate(mut member : Member, problem: &mut Problem) -> Member{
    // Mutation: Point-based mutation of sign or decimals
    let index = problem.rng.gen_range(0..member.len());
    let mut new_member = member.to_vec();
    if index == 0 {
        new_member[index] = num(0, 1, problem.rng);    
    } else {
        new_member[index] = num(problem.min, problem.max, problem.rng);
    }
    return new_member
}

fn crossover(a: Member, b: Member, problem : &mut Problem) -> 
            (Member, Member){
    // Point based crossover @ a random point
    // Generate two offspring from two parents
    // With a *single* point exchanged
    let point = problem.rng.gen_range(0..a.len()); // a as reference 
    let mut new_a = a.to_vec(); // Base on a
    let mut new_b = a.to_vec(); // Base on a
    new_a[point] = b[point];    
    new_b[point] = a[point];    
    return (new_a, new_b)
}

fn select(population : &mut Population, problem: &mut Problem) -> Metrics {
    // Issue: this calculates fitness twice, and I'm not good enough at
    // Rust to figure out exactly how to fix it.
    let mut fitnesses: Vec<Of64> = Vec::with_capacity(problem.pop_size);
    for i in 0..problem.pop_size {
        let f = (problem.fitness)(&population[i]);
        if f > problem.min_fit && 
           f < problem.max_fit {
        fitnesses.push(f);
       }
    }
    population.make_contiguous().sort_by_key(|m| (problem.fitness)(m));
    let avg = mean(&fitnesses);
    return Metrics{ min : fitnesses.iter().min().unwrap().clone(),
                    max : fitnesses.iter().max().unwrap().clone(),
                    avg : avg,
                    stdev : standard_deviation(&fitnesses, Some(avg))}
}


fn decode(x : &Member) -> f64 {
    // First, map x (vec in RN x {0, 1}) to x in R1
    let mut mapped : f64 = -1.0 * x[0] as f64;
    for i in 1..x.len() {
        mapped += x[i] as f64 / f64::powf(10., i as f64);
    }
    return mapped
}

fn problem_fitness(x : &Member) -> Of64 {
    // Assume x in [-0.5, 1]
    let mapped = decode(x);
    if mapped < -0.5 || mapped > 1.0 { 
        // Out of bounds, so use -∞ fitness
        // You would need to update this to make the fitness more generic
        return OrderedFloat(-1.0 * f64::INFINITY)
    }
    let loss: f64 = mapped * f64::sin(10.0 * std::f64::consts::PI * mapped) + 1.0;
    return OrderedFloat(loss)
}


pub fn simple_ga<'a>(iterations : u32) {

    let mut rng = rand::thread_rng();
    let mut problem = Problem{
        rng: &mut rng,
        min: 0,
        max: 9,
        length: 5,
        pop_size: 100,
        fitness: problem_fitness,
        minimizing: false,
        min_fit: OrderedFloat(-1.0 * f64::INFINITY),
        max_fit: OrderedFloat(f64::INFINITY)
    };

    let member = generate(&mut problem);
    println!("Member: {:#?}", &member);
    println!("Fitness: {}", (problem.fitness)(&member));
    let mut population = initialize(&mut problem);

    let metrics = select(&mut population, &mut problem);
    println!("Metrics: {:?}", metrics);

    // println!("Iterations: {}", iterations);
    // for i in 0..iterations {
    //     let (ai, bi) = top_two(&population, &mut problem);
    //     println!("Iteration: {}", i);
    //     println!{"Top two: {} {}", ai, bi}
    // }

    //for i in 0..iterations {
    //    let best_index = select(&population, &target, true);
    //    let copied  = population.get(best_index).clone().expect("Cannot clone").to_vec();
    //    let mutated = mutate(copied, &mut rng);
    //    population.push_back(mutated);
    //    population.pop_front();
    //}
}
