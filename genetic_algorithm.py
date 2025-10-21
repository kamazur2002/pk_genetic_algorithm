import numpy as np
from chromosome import Chromosome


class GeneticAlgorithm:
    """Main Genetic Algorithm implementation"""

    def __init__(self, config: dict):
        self._validate_config(config)
        self.config = config
        self.population = None

    def initialize_population(self) -> list:
        """Initialize population with random binary chromosomes"""
        population_size = self.config['population_size']
        n_variables = self.config['n_variables']
        bounds = self.config['bounds']
        precision = self.config['precision']

        population = []

        for _ in range(population_size):
            chromosome = self._generate_chromosome(
                n_variables, bounds, precision)
            population.append(chromosome)
        
        self.population = population
        return population

    def _calculate_gene_length(self, bound, precision):
        """Calculate the length of the gene for a given variable based on bounds and precision"""
        len_range = bound[1] - bound[0]
        m = np.ceil(np.log2(len_range * (10 ** precision)))
        return int(m)

    def _generate_chromosome(self, n_variables, bounds, precision):
        """Generate a random binary chromosome"""
        genes = []
        for i in range(n_variables):
            bound = bounds[i]
            m = self._calculate_gene_length(bound, precision)
            gene = ''.join(['1' if __import__('random').random()
                           > 0.5 else '0' for _ in range(m)])
            genes.append(gene)
        return Chromosome(genes, bounds, precision)

    def _validate_config(self, config: dict):
        """Validate configuration parameters"""
        required_keys = ['population_size',
                         'n_variables', 'bounds', 'precision']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config parameter: {key}")

        if not isinstance(config['population_size'], int) or config['population_size'] <= 0:
            raise ValueError("population_size must be a positive integer")

        if not isinstance(config['n_variables'], int) or config['n_variables'] <= 0:
            raise ValueError("n_variables must be a positive integer")

        if not isinstance(config['bounds'], list) or len(config['bounds']) != config['n_variables']:
            raise ValueError(
                "bounds must be a list with length equal to n_variables")

        for bound in config['bounds']:
            if (not isinstance(bound, tuple) or len(bound) != 2 or
                not all(isinstance(x, (int, float)) for x in bound) or
                    bound[0] >= bound[1]):
                raise ValueError(
                    "Each bound must be a tuple of two numbers (min, max) with min < max")

        if not isinstance(config['precision'], int) or config['precision'] <= 0:
            raise ValueError("precision must be a positive integer")
