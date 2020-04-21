import os
import sys

script_string = '''#!/bin/bash

# set the number of nodes and processes per node

#SBATCH --nodes={}

# set the number of tasks (processes) per node.
#SBATCH --ntasks-per-node={}

#SBATCH --partition=all
# set max wallclock time
#SBATCH --time=15:00:00

# set name of job
#SBATCH --job-name=mihailo-ilic-rvp-{}

mpirun -n {} build/program.o sonar.all-data 8 5000 12


'''

def main():

	script_name = "do_job"

	if not os.path.exists('./scripts'):
		os.mkdir('scripts')

	max_workers = int(sys.argv[1])
	# numNodes = sys.argv[2]

	# numWorkersPerNode = maxWorkers // numNodes

	num_workers_list = [1] + [n for n in range(5, max_workers + 1, 5)]

	print(num_workers_list)

	for n in num_workers_list:
		final_script = script_string.format(1, n, n, n)

		final_script_path = f'./scripts/{script_name}_{n}.sh'

		with open(final_script_path, "w") as f:
			f.write(final_script)

			command = f'''
			module load openmpi/3.1.3;
			module load openmpi/3.1.3-cuda
			sbatch {final_script_path}
			'''

			os.system(command)



if __name__ == '__main__':
	main()