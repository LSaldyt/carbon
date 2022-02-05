use rand::Rng;
use std::collections::VecDeque;
use std::assert;

type Member     = Vec<i32>; // In our problem, members are simple vectors
type Population = VecDeque<Member>; // Population is deque of members

struct Problem {
    rng: &mut rand::rngs::ThreadRng,
    min: i32,
    max: i32,
    length: usize,
    pop_size: usize,
    fitness: fn(&Member) -> f64,
    minimizing: bool,
}

fn num(min: i32, max: i32, rng : &mut rand::rngs::ThreadRng) -> i32 {
    // Silly convenience function because "gen_range"
    // is much longer to type than "num"
    return rng.gen_range(min..max+1);
}

fn generate(problem: Problem) -> Member {
    // Generate a random length l vector constrained to a range (min, max)
    // The first "bit" is reserved for sign
    let mut initial = vec![0; problem.length];
    initial[0] = num(0, 1, problem.rng);
    for i in 1..problem.length {
        initial[i] = num(problem.min, problem.max, problem.rng);
    }
    return initial
}

fn initialize(problem: Problem) -> Population {
    // Create the initial population
    let mut population: Population = VecDeque::with_capacity(problem.popsize);
    for _i in 0..problem.popsize {
        population.push_back(generate(problem));
    }
    return population
}

fn mutate(mut member : Member, problem: Problem) -> Member{
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

fn crossover(mut a : Member, mut b : Member, problem : Problem) -> 
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

fn top_two(population : &Population, 
           target : &Member, problem: Problem) -> (usize, usize) {
    assert!(population.len() > 1); // Need at least two members
    let (mut ai  , mut bi  ) = (0usize, 0usize);
    let (mut afit, mut bfit) = (0.0f64, 0.0f64);
    if problem.minimizing {
        afit = f64::INFINITY;
        bfit = f64::INFINITY;
    } else {
        afit = -1.0 * f64::INFINITY;
        bfit = -1.0 * f64::INFINITY;
    }

    for mi in 0..population.len() {
        let member = population.get(mi).expect("Logic Error");
        let fit: f64 = problem.fitness(&member, &target);
        if problem.minimizing {
            if fit < afit {
                afit = fit; ai = mi;
            } else if fit < bfit {
                bfit = fit; bi = mi;
            }
        } else {
            if fit > bfit {
                afit = fit; ai = mi;
            } else if fit > bfit {
                bfit = fit; bi = mi;
            }
        }
    }
    println!("Fitnesses: {}, {} ({}, {})", afit, bfit, ai, bi);
    return (ai, bi);
}

fn decode(x : &Member) -> f64 {
    // First, map x (vec in RN x {0, 1}) to x in R1
    let mut mapped : f64 = -1.0 * x[0] as f64;
    for i in 1..x.len() {
        mapped += x[i] as f64 / f64::pow(10., i);
    }
    return mapped
}

fn problem_fitness(x : &Member) -> f64 {
    // Assume x in [-0.5, 1]
    let mapped = decode(x);
    if mapped < -0.5 || mapped > 1.0 { 
        // Out of bounds, so use -∞ fitness
        return -1.0 * f64::INFINITY
    }
    let loss: f64 = mapped * f64::sin(10.0 * std::f64::consts::PI * mapped) + 1.0;
    return loss
}


pub fn simple_ga(iterations : u32) {

    //let mut rng = rand::thread_rng();
    let problem = Problem{
        rng: rand::thread_rng(),
        min: 0,
        max: 9,
        length: 5,
        pop_size: 10,
        fitness: problem_fitness,
        minimizing: true,
    };

    let member = generate(problem);
    println!("Member: {:#?}", &member);
    println!("Fitness: {}", problem.fitness(&member));
    let mut population = initialize(problem);

    //for i in 0..iterations {
    //    println!("Iteration: {}", i);
    //    let best_index = select(&population, &target, true);
    //    let copied  = population.get(best_index).clone().expect("Cannot clone").to_vec();
    //    let mutated = mutate(copied, &mut rng);
    //    population.push_back(mutated);
    //    population.pop_front();
    //}
}
