## Improved Sampling Algorithms for Statistical Physics
In this capstone project, I implemented the Metropolis-Hastings algorithm for the Ising model and for the Potts model. I also created functions for a modified version of the Metropolis-Hastings algorithm that incorporates a technique known as **landscape modification** developed by Professor Michael Choi. His paper directly relevant to these implementations can be found here: https://arxiv.org/abs/2011.09680. With my implementation, I find that landscape modification shows great potential in improving performance of Metropolis-Hastings algorithm for the Ising model and for the Potts model. In the visualizations section below, I display side-by-side visualizations of the algorithms over time and show their energy plots contrasted against each other. 

## Acknowledgements
The code for my implementation, in particular the functions to generate random lattice configurations and calculate the configuration energy for the Ising model, is modified from work by Tanya Schlusser on GitHub. The reference code can be referred to here: https://tanyaschlusser.github.io/posts/mcmcand-the-ising-model/.

## Visualizations

### Ising model: Original MH vs Quadratic MH using a fixed value of c vs Quadratic MH using adaptive piecewise c
![66origvsquadtypes](https://github.com/nicolecaps/capstone/assets/111272955/c72e1d0b-efc9-47f1-8561-c2db3414e95a)

### Ising model: Original MH vs Linear MH vs Quadratic MH vs Square Root MH (all variants using adaptive piecewise c)
![66origvsadaptive](https://github.com/nicolecaps/capstone/assets/111272955/7fe213b9-b8bf-4069-90c0-294ec0660480)

### 3-state Potts model Example 1 (Original MH stuck in local minima, variants able to escape local minima)
![pottsAP-seed475](https://github.com/nicolecaps/capstone/assets/111272955/bc5d4be7-a4ac-4f89-ad92-5fe91eea3e37)

### 3-state Potts model Example 2 (variants escaping local minima and reaching different global minima)
![pottsexample475 ](https://github.com/nicolecaps/capstone/assets/111272955/6216ddba-8f02-4b8b-a01b-d0bd0178bc33)
