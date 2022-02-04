use rand::Rng;
use std::collections::VecDeque;
use std::assert;
use itertools::izip;

const MAXC: i32 = 100;
const MINC: i32 = -100;

fn num(rng : &mut rand::rngs::ThreadRng) -> i32 {
    return rng.gen_range(MINC..MAXC+1);
}

fn generate(length : usize, rng : &mut rand::rngs::ThreadRng) -> Vec<i32> {
    let mut initial = vec![0; length];
    for i in 0..length {
        initial[i] = num(rng);
    }
    return initial
}

fn initialize(popsize : usize, 
              member_length : usize, 
              rng : &mut rand::rngs::ThreadRng) -> VecDeque<Vec<i32>> {
    let mut population: VecDeque<Vec<i32>> = VecDeque::with_capacity(popsize);
    for _i in 0..popsize {
        population.push_back(generate(member_length, rng));
    }
    return population
}

fn mutate(mut member : Vec<i32>, rng : &mut rand::rngs::ThreadRng) -> Vec<i32>{
    let index = rng.gen_range(0..member.len());
    let new = num(rng);
    member[index] = new; // Random sample from [MINC, MAXC]
    return member
}

fn fitness(a : &Vec<i32>, b : &Vec<i32>) -> f64 {
    let mut sum : f64 = 0.0;
    assert!(a.len() == b.len());
    for (ai, bi) in izip!(a, b) {
        sum = sum + ((ai - bi).pow(2) as f64);
    }
    return sum;
}

fn select(population : &VecDeque<Vec<i32>>, 
          target : &Vec<i32>, 
          minimizing : bool) -> usize {
    assert!(population.len() > 0);
    let mut index : usize = 0;
    let mut best  : f64;
    if minimizing {
        best = f64::INFINITY;
    } else {
        best = -1.0 * f64::INFINITY;
    }

    for mi in 0..population.len() {
        let member = population.get(mi).expect("Logic Error");
        let fit : f64 = fitness(&member, &target);
        if minimizing {
            if fit < best {
                best = fit;
                index = mi;
            } 
        } else {
            if fit > best {
                best = fit;
                index = mi;
            }
        }
    }
    println!("Fitness:   {}", best);
    return index;
}

pub fn regev(iterations : u32) {
    let member_length : usize = 1000;
    let mut rng = rand::thread_rng();
    let target = generate(member_length, &mut rng);
    // println!("Target: {:?}", target);
    let test = generate(member_length, &mut rng);
    // println!("Test: {:?}", test);
    let fit : f64 = fitness(&test, &target);
    println!("Fitness: {}", fit);
    let mut population = initialize(1000, member_length, &mut rng);

    for i in 0..iterations {
        println!("Iteration: {}", i);
        let best_index = select(&population, &target, true);
        let copied  = population.get(best_index).clone().expect("Cannot clone").to_vec();
        let mutated = mutate(copied, &mut rng);
        population.push_back(mutated);
        population.pop_front();
    }
}
