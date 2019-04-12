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
}

data {
  int<lower=2> T;                   // number of capture events (includes marking)
  int<lower=0> M;                   // number of unique capture histories
  int<lower=1> K;                   // total number of covariates
  matrix[M,K] X;                    // covariates (first column is 1 for intercept)
  int<lower=0,upper=1> indX_phi[K,T-1]; // use covariate k for phi[t]?
  int<lower=1> group_phi[M,T-1];    // phi group IDs for each unique capture history
  vector<lower=0,upper=1>[M] fix_phi0[T-1]; // fix phi[m,t] to zero?
  int<lower=0,upper=1> indX_p[K,T]; // use covariate k for p[t]?
  int<lower=1> group_p[M,T];        // p group IDs for each unique capture history
  vector<lower=0,upper=1>[M] fix_p1[T]; // fix p[m,t] to 1?
  int<lower=0,upper=1> y[M,T];      // y[m,t]: history m captured at t
  int<lower=1> n[M];                // n[m]: number of individuals with capture history y[m,]
}

transformed data {
  int<lower=1> K_phi;                   // number of covariates for phi
  int<lower=1> K_p;                     // number of covariates for p
  int<lower=1> J_phi;                   // number of groups for phi
  int<lower=1> J_p;                     // number of groups for p
  vector<lower=0,upper=1>[T-1] random_phi; // include random effects for phi[t]?
  vector<lower=0,upper=1>[T] random_p;     // include random effects for p[t]?
  int<lower=0,upper=T> first[M];        // first capture occasion
  int<lower=0,upper=T> last[M];         // last capture occasion
  int<lower=0,upper=T-1> last_minus_first[M];  // duh
  
  K_phi = sum(to_array_1d(indX_phi));
  K_p = sum(to_array_1d(indX_p));
  J_phi = max(to_array_1d(group_phi));
  J_p = max(to_array_1d(group_p));
  
  // If only one group for a particular phi[t] or p[t], don't use random effects
  for(t in 1:(T-1))
    random_phi[t] = min(group_phi[,t]) < max(group_phi[,t]);
  for(t in 1:T)
    random_p[t] = min(group_p[,t]) < max(group_p[,t]);
  
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
  // Fix phi to zero and p to 1 where specified
  for(t in 1:(T-1))
    phi[,t] = inv_logit(X * beta[,t] + random_phi[t] * sigma[t] * zeta[group_phi[,t],t]) .* (1 - fix_phi0[t]);

  for(t in 1:T)
    p[,t] = inv_logit(X * b[,t] + random_p[t] * s[t] * z[group_p[,t],t]) .* (1 - fix_p1[t]) + fix_p1[t];

  // Likelihood of capture history
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


