# Masters\_CE
Generalized Next Generation Reservoir Computing (NGRC)

## working documentation
- Execute NGRC program with ./scripts/run from root directory
- Configuration files can be found in configs directory 
- Program results can be found in results/output.txt and results/plot.png
- Additional dynamical systems can be added by modifying src/data\_generation.py file. Template can be found at top of file.
- Script for calculating Lyapunov exponents/time in scripts/lyapunov\_time. 
    - Runs src/layapunov\_time/find\_lyapunov\_time.py, with systems being defined in src/layapunov\_time/systems.py

## NGRC overview
0. Configuration File/ scripts
1. data generation
2. feature vector construction
3. perform regression
4. calculate training fit NRMSE
5. make prediction
6. calculate prediction NRMSE

## 0. Configuration File
Parameters:

## 1. Data Generation
- self implemented runge-kutta-4
- scipy solve\_ivp solvers
- TODO: upload data csv file

## 2. Feature Vector Construction
- construct feature vector with parameters s, k, p.
- p is plynomial power of feature vector ($p \leq 9$)

## 3. Regression methods
This program can use Ridge, Lasso, and Elastic Net regression methods

### Ridge
```math
\min_{\beta} \{  || y-X \beta ||_2^2 + \lambda_2 || \beta ||_2^2  \}
```

### Lasso
```math
\min_{\beta} \{  || y-X \beta ||_2^2 + \lambda_1 || \beta ||_1  \}
```
### Elastic Net
```math
\min_{\beta} \{  || y-X \beta ||_2^2 + \lambda_1 || \beta ||_1 + \lambda_2 || \beta ||_2^2  \}
```
