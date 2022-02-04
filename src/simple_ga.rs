use rand::Rng;
use std::collections::VecDeque;
use std::assert;

fn num(min: i32, max: i32, rng : &mut rand::rngs::ThreadRng) -> i32 {
    return rng.gen_range(min..max+1);
}

fn generate(length : usize, min: i32, max: i32, 
            rng : &mut rand::rngs::ThreadRng) -> Vec<i32> {
    // Generate a random length l vector constrained to a range (min, max)
    // The first "bit" is reserved for sign
    let mut initial = vec![0; length];
    initial[0] = num(0, 1, rng);
    for i in 1..length {
        initial[i] = num(min, max, rng);
    }
    return initial
}

fn initialize(popsize : usize, member_length : usize, min : i32, max : i32,
              rng : &mut rand::rngs::ThreadRng) -> VecDeque<Vec<i32>> {
    // Create the initial population
    let mut population: VecDeque<Vec<i32>> = VecDeque::with_capacity(popsize);
    for _i in 0..popsize {
        population.push_back(generate(member_length, min, max, rng));
    }
    return population
}

fn mutate(mut member : Vec<i32>, min:i32, max:i32, rng : &mut rand::rngs::ThreadRng) -> Vec<i32>{
    // Mutation: currently empty
    let index = rng.gen_range(0..member.len());
    let mut new_member = member.to_vec();
    if index == 0 {
        new_member[index] = num(0, 1, rng);    
    } else {
        new_member[index] = num(min, max, rng);
    }
    return new_member
}

fn fitness(x : &Vec<i32>) -> f64 {
    // Assume x in [-0.5, 1]
    // First, map x (vec in R4 x {0, 1}) to x in R1
    let mapped : f64 = -1.0 * x[0] as f64 + 
                       x[1] as f64 / 10. + 
                       x[2] as f64 / 100. + 
                       x[3] as f64 / 1000. + 
                       x[4] as f64 / 10000.;
    let loss : f64 = mapped * f64::sin(10.0 * std::f64::consts::PI * mapped) + 1.0;
    return loss
}

fn select(population : &VecDeque<Vec<i32>>, 
          target : &Vec<i32>, 
          minimizing : bool) -> usize {
    assert!(population.len() > 0);
    return 0;
    //let mut index : usize = 0;
    //let mut best  : f64;
    //if minimizing {
    //    best = f64::INFINITY;
    //} else {
    //    best = -1.0 * f64::INFINITY;
    //}

    //for mi in 0..population.len() {
    //    let member = population.get(mi).expect("Logic Error");
    //    let fit : f64 = fitness(&member, &target);
    //    if minimizing {
    //        if fit < best {
    //            best = fit;
    //            index = mi;
    //        } 
    //    } else {
    //        if fit > best {
    //            best = fit;
    //            index = mi;
    //        }
    //    }
    //}
    //println!("Fitness:   {}", best);
    //return index;
}

pub fn simple_ga(iterations : u32) {

    let mut rng = rand::thread_rng();
    let member = generate(5, 0, 9, &mut rng);
    println!("Member: {:#?}", &member);
    println!("Fitness: {}", fitness(&member));
    //let member_length : usize = 1000;
    //let target = generate(member_length, &mut rng);
    //// println!("Target: {:?}", target);
    //let test = generate(member_length, &mut rng);
    //// println!("Test: {:?}", test);
    //let fit : f64 = fitness(&test, &target);
    //println!("Fitness: {}", fit);
    //let mut population = initialize(1000, member_length, &mut rng);

    //for i in 0..iterations {
    //    println!("Iteration: {}", i);
    //    let best_index = select(&population, &target, true);
    //    let copied  = population.get(best_index).clone().expect("Cannot clone").to_vec();
    //    let mutated = mutate(copied, &mut rng);
    //    population.push_back(mutated);
    //    population.pop_front();
    //}
}
