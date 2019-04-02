// Cormack-Jolly-Seber Model (m-array data format) with covariates and
// random effects on time-specific survival and capture probabilities 
// (possibly different grouping variables for each phi and p)
// as well as a code that indicates phi = 0, p = 1

functions {
  int first_capture(int[] y_i) {
    for (t in 1:size(y_i))
      if (y_i[t])
        return t;
    return 0;
  }
  
  int last_capture(int[] y_i) {
    for (t_rev in 0:(size(y_i) - 1)) 
    {
      int t = size(y_i) - t_rev;
      if (y_i[t])
        return t;
    }
    return 0;
  }
  
  real prob_uncaptured(int last_capture, row_vector p, row_vector phi) {
    int T = num_elements(p);
    row_vector[T] chi;
    
    chi[T] = 1.0;
    for (t in 1:(T - last_capture)) 
    {
      int t_curr = T - t;
      int t_next = t_curr + 1;
      chi[t_curr] = (1 - phi[t_curr]) + phi[t_curr] * (1 - p[t_next]) * chi[t_next];
    }
    return chi[last_capture];
  }

  // row_vector prob_uncaptured(int T, row_vector p, row_vector phi) {
  //   row_vector[T] chi;
  //   
  //   chi[T] = 1.0;
  //   for (t in 1:(T - 1)) 
  //   {
  //     int t_curr = T - t;
  //     int t_next = t_curr + 1;
  //     chi[t_curr] = (1 - phi[t_curr]) + phi[t_curr] * (1 - p[t_next]) * chi[t_next];
  //   }
  //   return chi;
  // }
}

data {
  int<lower=2> T;                   // number of capture events (includes marking)
  int<lower=0> M;                   // number of unique capture histories
  int<lower=1> K;                   // total number of covariates
  matrix[M,K] X;                    // covariates (first column is 1 for intercept)
  int<lower=0,upper=1> indX_phi[K,T-1]; // use covariate k for phi[t]?
  int<lower=0> group_phi[M,T-1];    // phi group IDs for each unique capture history
  int<lower=0,upper=1> indX_p[K,T]; // use covariate k for p[t]?
  int<lower=0> group_p[M,T];        // p group IDs for each unique capture history
  int<lower=0,upper=1> y[M,T];      // y[m,t]: history m captured at t
  int<lower=1> n[M];                // n[m]: number of individuals with capture history y[m,]
}

transformed data {
  int<lower=1> K_phi;                   // number of covariates for phi
  int<lower=1> K_p;                     // number of covariates for p
  int<lower=1> J_phi;                   // number of groups for phi
  int<lower=1> J_p;                     // number of groups for p
  int<lower=0,upper=1> random_phi[T-1]; // include random effects for phi[t]?
  int<lower=0,upper=1> random_p[T];     // include random effects for p[t]?
  int<lower=0,upper=T> first[M];        // first capture occasion
  int<lower=0,upper=T> last[M];         // last capture occasion
  int<lower=0,upper=T-1> last_minus_first[M];  // duh
  
  K_phi = sum(to_array_1d(indX_phi));
  K_p = sum(to_array_1d(indX_p));
  J_phi = max(to_array_1d(group_phi));
  J_p = max(to_array_1d(group_p));
  
  // If only one group for a particular phi[t] or p[t], don't use random effects
  for(t in 1:(T-1))
    random_phi[t] = min(group_phi[,t]) == max(group_phi[,t]);
  for(t in 1:T)
    random_p[t] = min(group_p[,t]) == max(group_p[,t]);
    
  
  for (m in 1:M)
  {
    first[m] = first_capture(y[m,]);
    last[m] = last_capture(y[m,]);
    last_minus_first[m] = last[m] - first[m];
  }
}

parameters {
  vector[K_phi] beta_vec;        // regression coefficients for logit(phi)
  vector<lower=0>[T-1] sigma;    // among-group SDs of logit(phi[,t])
  matrix[J_phi,T-1] zeta;        // group-specific random effects on phi (z-scores)
  vector[K_p] b_vec;             // regression coefficients for logit(p)
  vector<lower=0>[T] s;          // among-group SDs of logit(p[,t])
  matrix[J_p,T] z;               // group-specific random effects on p (z-scores)
}

transformed parameters {
  matrix[K,T-1] beta;   // regression coefficients for logit(phi) with structural zeros
  matrix[K,T] b;        // regression coefficients for logit(p) with structural zeros
  matrix[M,T-1] phi;    // phi[,t]: Pr[alive at t + 1 | alive at t]
  matrix[M,T] p;        // p[,t]: Pr[captured at t | alive at t] (note p[,1] not used in model)
  vector[M] chi;        // chi[m]: Pr[not captured >  last[m] | alive at last[m]]
  // matrix[M,T] chi;      // chi[,t]: Pr[not captured >  t | alive at t]
  vector[M] LL;         // log-likelihood of each capture history
  
  // Fill in sparse beta and b matrices
  beta = rep_matrix(0, K, T-1);
  b = rep_matrix(0, K, T);
  
  {
    int np_phi;
    int np_p;
    
    np_phi = 1;
    np_p = 1;
    
    for(k in 1:K)
    {
      for(t in 1:(T-1))
      if(indX_phi[k,t])
      {
        beta[k,t] = beta_vec[np_phi];
        np_phi += 1;
      }
      
      for(t in 1:T)
      if(indX_p[k,t])
      {
        b[k,t] = b_vec[np_p];
        np_p += 1;
      }
    }
  }
  
  // Hierarchical logistic regression for phi and p
  for(t in 1:(T-1))
  {
    // if(group_phi[m,t] == 0)  // special code: fix survival to 0 and detection to 1
    //   phi[m,t] = 0;
    if(random_phi[t])
      phi[,t] = inv_logit(X * beta[,t] + sigma[t] * zeta[group_phi[,t],t]);
    else
      phi[,t] = inv_logit(X * beta[,t]);
  }
  
  for(t in 1:T)
  {
    // if(group_p[m,t] == 0)
    //   p[m,t] = 1;
    if(random_p[t])
      p[,t] = inv_logit(X * b[,t] + s[t] * z[group_p[,t],t]);
    else
      p[,t] = inv_logit(X * b[,t]);
  }
  
  // for(m in 1:M)
  // {
  //   for(t in 1:(T-1))
  //   {
  //     if(group_phi[m,t] == 0)  // special code: fix survival to 0 and detection to 1
  //       phi[m,t] = 0;
  //     else if(random_phi[t])
  //       phi[m,t] = inv_logit(X[m,] * beta[,t] + sigma[t] * zeta[group_phi[m,t],t]);
  //     else
  //       phi[m,t] = inv_logit(X[m,] * beta[,t]);
  //   }
  // 
  //   for(t in 1:T)
  //   {
  //     if(group_p[m,t] == 0)
  //       p[m,t] = 1;
  //     else if(random_p[t])
  //       p[m,t] = inv_logit(X[m,] * b[,t] + s[t] * z[group_p[m,t],t]);
  //     else
  //       p[m,t] = inv_logit(X[m,] * b[,t]);
  //   }
  // }
  
  // Likelihood of capture history, marginalized over discrete latent states
  LL = rep_vector(0,M);
  
  for(m in 1:M) 
  {
    if (last_minus_first[m] > 0)  // if history m was recaptured
    {
      for(t in (first[m]+1):last[m])
      {
        LL[m] += n[m] * log(phi[m,t-1]);                 // survival from t - 1 to t
        LL[m] += n[m] * bernoulli_lpmf(y[m,t] | p[m,t]); // observation (captured or not)
      }
    }
    chi[m] = prob_uncaptured(last[m], p[m,], phi[m,]);
    LL[m] += n[m] * log(chi[m]);   // Pr[not detected after last[m]]
    // chi[m,] = prob_uncaptured(T, p[m,], phi[m,]);
    // LL[m] += n[m] * log(chi[m,last[m]]);   // Pr[not detected after last[m]]
  }
}

model {
  // Priors 
  
  // log Jacobian of logit transform for phi[t] intercepts
  // implies phi[t] ~ Unif(0,1) given all covariates are at their sample means
  target += log_inv_logit(beta_vec[1:(T-1)]) + log1m_inv_logit(beta_vec[1:(T-1)]);
  if(K_phi > T - 1)
    beta_vec[T:K_phi] ~ normal(0,3); 
  sigma ~ normal(0,3);    
  to_vector(zeta) ~ normal(0,1);  // implies logit(phi[m,t]) ~ N(logit(mu_phi[t]), sigma);
  // log Jacobian of logit transform for p[t] intercepts
  // implies p[t] ~ Unif(0,1) given all covariates are at their sample means
  target += log_inv_logit(b_vec[1:T]) + log1m_inv_logit(b_vec[1:T]);
  if(K_p > T)
    b_vec[(T+1):K_p] ~ normal(0,3);     
  s ~ normal(0,3); 
  to_vector(z) ~ normal(0,1);    // implies logit(p[m,t]) ~ N(logit(mu_p[t]), s);
  
  // Likelihood of capture history added to log posterior
  target += sum(LL);
}

generated quantities {
  matrix[J_phi,T-1] epsilon;     // group-specific random effects on phi
  matrix[J_p,T] e;               // group-specific random effects on p
  
  epsilon = diag_post_multiply(zeta, sigma);
  e = diag_post_multiply(z, s);
}


