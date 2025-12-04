# Masters\_CE
Generalized Next Generation Reservoir Computing (NGRC)

## working documentation
- Execute program with ./scripts/run from root directory
- Configuration files can be found in configs directory 
- Program results can be found in results/output.txt and results/plot.png
- Additional dynamical systems can be added by modifying src/data\_generation.py file. Template can be found at top of file.

## NGRC overview
1. data generation
2. feature vector construction
3. perform regression
4. calculate training fit NRMSE
5. make prediction
6. calculate prediction NRMSE

## 1. Data Generation

## 2. Feature Vector Construction

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
## Configuration File
