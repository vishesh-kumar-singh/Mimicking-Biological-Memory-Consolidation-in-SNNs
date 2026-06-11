from scripts.run_phase_b_p_factor import run_experiment as run_p
from scripts.run_freezing import run_experiment as run_f

h_p = run_p(0, 1, 42, 0.2)
h_f = run_f(0, 1, 42, 0.2)
print("P:", h_p['final_task_a'])
print("F:", h_f['final_task_a'])
