# CBF_Engine
![CBF Engine Test](https://andrewadie.com/wp-content/uploads/2026/05/cbf_engine_test.gif))

## Project Summary
Python prototype for C++ Dynamic Exponential Control Barrier Function engine. The intent of this tool is to easily create and specify several control barriers in a problem space. Trajectories which want to use this engine should use CVXP. This engine is built with jax for future optimization. It is designed to be a light-weight obstacle avoidance simulator for trajectory simulation.

## Project History
Mk I - Built CBF engine for static and dynamic obstacles with memory management, circular obstacles, and discrete exponential control barrier function (DECBF) models.


## Mathematical Derivation
$$ h(p_{t+1}, t+1) $$

$$ Gp_t \ge (1 - \gamma) * h(p_t) $$

$$ \gamma \in (0,1]$$

Taylor Expansion:

$$h(p_{t+1}, t+1) \approx h(p_t,t) + \frac{\partial h}{\partial p}(p_{t+1}-p_t) + \frac{\partial h}{\partial t} dt $$

By Substitution:

$$ \frac{\partial h}{\partial p} p_{t+1} + (\gamma h(p_t,t) - \frac{\partial h}{\partial p}p_t + \frac{\partial h}{\partial t}dt) \ge 0 $$

Using the gradient defined above:

$$ G = 2(p_t - p_{obs,t})^T $$

Applyng chain rule to temporal element:

$$ \frac{\partial h}{\partial t} = \frac{\partial h}{\partial p_{obs}} \frac{\partial p_{obs}}{\partial t} = -Gv_{obs}$$

$$ \alpha * dt = \gamma $$

Therefore:

$$ \boxed{ Gp_{t+1} + (\gamma h(p_t, t) - Gp_t - Gv_{obs}dt) \ge 0 } $$

Future Improvements:
- Incorporate better dynamics models
- Improve vectorization
- Capitalize on JAX library
- Expand Ellipse Geometry class beyond circles



