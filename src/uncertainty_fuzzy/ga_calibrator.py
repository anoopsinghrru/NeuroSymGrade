"""
Numba-Accelerated Genetic Algorithm Calibrator with Delta-Encoding (Unit 6).
Tunes fuzzy membership boundaries against gold-standard human-graded samples.
Guarantees a < b < c via delta-encoded chromosomes [a, Delta_1, Delta_2].
"""

import numpy as np
from typing import Tuple, Optional

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@njit(fastmath=True)
def decode_delta_chromosome(chromosome: np.ndarray) -> np.ndarray:
    """
    Decodes an 18-element delta-chromosome into standard [a, b, c] vertex parameters.
    chromosome: 6 groups of [a_k, Delta_1_k, Delta_2_k]
    decoded: 6 groups of [a_k, a_k + Delta_1, a_k + Delta_1 + Delta_2]
    Guarantees a < b < c strictly without needing repair loops!
    """
    decoded = np.zeros(18, dtype=np.float64)
    for k in range(6):
        idx = k * 3
        a = chromosome[idx]
        delta_1 = max(chromosome[idx + 1], 0.01)
        delta_2 = max(chromosome[idx + 2], 0.01)

        b = a + delta_1
        c = b + delta_2

        decoded[idx] = a
        decoded[idx + 1] = b
        decoded[idx + 2] = c

    return decoded


@njit(fastmath=True)
def eval_trimf(x: float, a: float, b: float, c: float) -> float:
    """Fast JIT-compiled triangular membership evaluation."""
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a) if b > a else 1.0
    else:
        return (c - x) / (c - b) if c > b else 1.0


@njit(fastmath=True)
def simulate_fuzzy_batch(
    params: np.ndarray,
    inputs: np.ndarray,  # Shape (N, 3): [defeasible_mult, belief, local_res]
    max_marks: np.ndarray # Shape (N,)
) -> np.ndarray:
    """Fast vectorized simulation of fuzzy evaluations for N samples."""
    n = inputs.shape[0]
    scores = np.zeros(n, dtype=np.float64)
    z_grid = np.linspace(0.0, 1.0, 51)

    for i in range(n):
        mult = inputs[i, 0]
        if mult == 0.0:
            scores[i] = 0.0
            continue

        belief = inputs[i, 1]
        local_res = inputs[i, 2]
        res_factor = max(0.0, min(1.0, 1.0 - (local_res / 5.0)))

        x_in = (0.60 * mult) + (0.30 * belief) + (0.10 * res_factor)
        x_in = max(0.0, min(1.0, x_in))

        mu_low = eval_trimf(x_in, params[0], params[1], params[2])
        mu_med = eval_trimf(x_in, params[3], params[4], params[5])
        mu_high = eval_trimf(x_in, params[6], params[7], params[8])

        num = 0.0
        den = 0.0
        for z in z_grid:
            w1 = min(mu_low, eval_trimf(z, params[9], params[10], params[11]))
            w2 = min(mu_med, eval_trimf(z, params[12], params[13], params[14]))
            w3 = min(mu_high, eval_trimf(z, params[15], params[16], params[17]))
            mu_agg = max(w1, w2, w3)

            num += z * mu_agg
            den += mu_agg

        z_star = (num / den) if den > 1e-6 else x_in
        scores[i] = z_star * max_marks[i]

    return scores


@njit(fastmath=True)
def compute_population_fitness(
    population: np.ndarray, # Shape (PopSize, 18)
    inputs: np.ndarray,
    max_marks: np.ndarray,
    gold_scores: np.ndarray,
    prior_params: np.ndarray,
    lambda_reg: float
) -> np.ndarray:
    """Computes fitness for every individual in the population."""
    pop_size = population.shape[0]
    n_samples = inputs.shape[0]
    fitnesses = np.zeros(pop_size, dtype=np.float64)

    for p in range(pop_size):
        decoded = decode_delta_chromosome(population[p])
        sim_scores = simulate_fuzzy_batch(decoded, inputs, max_marks)

        # Mean Squared Error (MSE)
        mse = 0.0
        for i in range(n_samples):
            diff = sim_scores[i] - gold_scores[i]
            mse += diff * diff
        mse /= n_samples

        # L2 Regularization penalty from prior thresholds to avoid overfitting
        l2 = 0.0
        for j in range(18):
            d = decoded[j] - prior_params[j]
            l2 += d * d

        fitness = 1.0 / (1.0 + mse + (lambda_reg * l2))
        fitnesses[p] = fitness

    return fitnesses


class GeneticAlgorithmCalibrator:
    """
    Genetic Algorithm optimizer for tuning Fuzzy Logic membership parameters.
    Uses delta-encoding to guarantee validity of fuzzy triangles.
    """

    def __init__(
        self,
        pop_size: int = 40,
        generations: int = 50,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.80,
        lambda_reg: float = 0.005
    ):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.lambda_reg = lambda_reg

        # Expert prior parameters
        self.prior_params = np.array([
            0.0, 0.25, 0.50,
            0.30, 0.60, 0.90,
            0.70, 0.90, 1.00,
            0.0, 0.15, 0.40,
            0.30, 0.60, 0.85,
            0.75, 0.95, 1.00
        ], dtype=np.float64)

    def generate_synthetic_gold_dataset(self, n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates realistic synthetic gold-standard human graded derivation samples.
        Inputs: [defeasible_mult, belief, local_res]
        """
        np.random.seed(42)
        inputs = np.zeros((n_samples, 3), dtype=np.float64)
        max_marks = np.full(n_samples, 10.0, dtype=np.float64)
        gold_scores = np.zeros(n_samples, dtype=np.float64)

        for i in range(n_samples):
            # Mix of full credit, ECF, rounding, failed, and veto
            r = np.random.rand()
            if r < 0.35: # Full credit
                mult = 1.0
                belief = np.random.uniform(0.85, 0.98)
                res = 0.0
                gold = 10.0
            elif r < 0.65: # ECF override
                mult = 0.85
                belief = np.random.uniform(0.75, 0.90)
                res = 0.0
                gold = 8.5
            elif r < 0.80: # Rounding override
                mult = 0.95
                belief = np.random.uniform(0.80, 0.95)
                res = np.random.uniform(0.01, 0.04)
                gold = 9.5
            elif r < 0.95: # Calculation slip
                mult = 0.25
                belief = np.random.uniform(0.60, 0.85)
                res = np.random.uniform(2.0, 8.0)
                gold = 2.5
            else: # Symbolic veto
                mult = 0.0
                belief = np.random.uniform(0.1, 0.4)
                res = 10.0
                gold = 0.0

            inputs[i] = [mult, belief, res]
            gold_scores[i] = gold

        return inputs, max_marks, gold_scores

    def initialize_delta_population(self) -> np.ndarray:
        """Initializes population around prior parameters encoded as deltas."""
        pop = np.zeros((self.pop_size, 18), dtype=np.float64)
        
        # Convert prior_params to base delta chromosome
        base_delta = np.zeros(18, dtype=np.float64)
        for k in range(6):
            idx = k * 3
            a = self.prior_params[idx]
            b = self.prior_params[idx + 1]
            c = self.prior_params[idx + 2]
            base_delta[idx] = a
            base_delta[idx + 1] = max(b - a, 0.05)
            base_delta[idx + 2] = max(c - b, 0.05)

        for p in range(self.pop_size):
            noise = np.random.normal(0, 0.02, 18)
            cand = base_delta + noise
            # Bound deltas to be positive
            for k in range(6):
                cand[k * 3 + 1] = max(cand[k * 3 + 1], 0.02)
                cand[k * 3 + 2] = max(cand[k * 3 + 2], 0.02)
            pop[p] = cand

        return pop

    def run_calibration(
        self,
        inputs: Optional[np.ndarray] = None,
        max_marks: Optional[np.ndarray] = None,
        gold_scores: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Executes Genetic Algorithm evolution across generations.
        Returns: (optimal_decoded_params, fitness_history).
        """
        if inputs is None or max_marks is None or gold_scores is None:
            inputs, max_marks, gold_scores = self.generate_synthetic_gold_dataset()

        population = self.initialize_delta_population()
        fitness_history = []

        for gen in range(self.generations):
            fitnesses = compute_population_fitness(
                population, inputs, max_marks, gold_scores, self.prior_params, self.lambda_reg
            )
            best_idx = np.argmax(fitnesses)
            best_fit = fitnesses[best_idx]
            fitness_history.append(float(best_fit))

            # Selection: Tournament
            new_pop = np.zeros_like(population)
            new_pop[0] = population[best_idx] # Elitism

            for i in range(1, self.pop_size):
                # Tournament of size 3
                t1, t2, t3 = np.random.randint(0, self.pop_size, 3)
                p1 = population[t1] if fitnesses[t1] >= fitnesses[t2] else population[t2]
                parent1 = p1 if fitnesses[t1] >= fitnesses[t3] else population[t3]

                t4, t5, t6 = np.random.randint(0, self.pop_size, 3)
                p2 = population[t4] if fitnesses[t4] >= fitnesses[t5] else population[t5]
                parent2 = p2 if fitnesses[t4] >= fitnesses[t6] else population[t6]

                # Uniform / Arithmetic Crossover
                if np.random.rand() < self.crossover_rate:
                    alpha = np.random.uniform(0.2, 0.8)
                    child = (alpha * parent1) + ((1.0 - alpha) * parent2)
                else:
                    child = parent1.copy()

                # Gaussian Mutation with delta preservation
                if np.random.rand() < self.mutation_rate:
                    mut_idx = np.random.randint(0, 18)
                    child[mut_idx] += np.random.normal(0, 0.03)

                # Safeguard: enforce non-negative deltas
                for k in range(6):
                    child[k * 3 + 1] = max(child[k * 3 + 1], 0.01)
                    child[k * 3 + 2] = max(child[k * 3 + 2], 0.01)

                new_pop[i] = child

            population = new_pop

        # Final evaluation
        final_fits = compute_population_fitness(
            population, inputs, max_marks, gold_scores, self.prior_params, self.lambda_reg
        )
        best_chrom = population[np.argmax(final_fits)]
        optimal_params = decode_delta_chromosome(best_chrom)

        return optimal_params, fitness_history
