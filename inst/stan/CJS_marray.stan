// Cormack-Jolly-Seber Model (m-array data format)

functions {
  int first_capture(int[] y_i) {
    for (t in 1:size(y_i))
      if(y_i[t])
        return t;
    return 0;
  }
  
  int last_capture(int[] y_i) {
    for (t_rev in 0:(size(y_i) - 1)) 
    {
      int t;
      t = size(y_i) - t_rev;
      if(y_i[t])
        return t;
    }
    return 0;
  }
  
  vector prob_uncaptured(int T, vector p, vector phi) {
    vector[T] chi;
    
    chi[T] = 1.0;
    for (t in 1:(T - 1)) 
    {
      int t_curr;
      int t_next;
      t_curr = T - t;
      t_next = t_curr + 1;
      chi[t_curr] = (1 - phi[t_curr]) + phi[t_curr] * (1 - p[t_next]) * chi[t_next];
    }
    return chi;
  }
}

data {
  int<lower=2> T;                  // number of capture events (includes marking)
  int<lower=0> M;                  // number of unique capture histories
  int<lower=0,upper=1> y[M,T];     // y[m,t]: history m captured at t
  int<lower=1> n[M];               // n[m]: number of individuals with capture history y[m,]
}

transformed data {
  int<lower=0,upper=T> first[M];   // first capture occasion
  int<lower=0,upper=T> last[M];    // last capture occasion
  int<lower=0,upper=T-1> last_minus_first[M];  // duh
  
  for (m in 1:M)
  {
    first[m] = first_capture(y[m,]);
    last[m] = last_capture(y[m,]);
    last_minus_first[m] = last[m] - first[m];
  }
}

parameters {
  vector<lower=0,upper=1>[T-1] phi;     // survival probabilities
  vector<lower=0,upper=1>[T] p;         // capture probabilities
}

transformed parameters {
  vector<lower=0,upper=1>[T] chi;       // chi[,t]: Pr[not captured >  t | alive at t]

  chi = prob_uncaptured(T, p, phi);
}

model {
  // implied uniform priors:
  // phi ~ uniform(0,1)
  // p ~ uniform(0,1)
  
  // Likelihood of capture history
  // marginalized over discrete latent states
  for (m in 1:M) 
  {
    if (last_minus_first[m] > 0)  // if history m was recaptured
    {
      for(t in (first[m]+1):last[m])
      {
        target += n[m] * log(phi[t-1]);                 // survival from t - 1 to t
        target += n[m] * bernoulli_lpmf(y[m,t] | p[t]); // observation (captured or not)
      }
    }
    target += n[m] * log(chi[last[m]]); // Pr[not detected after last[m]]
  }
}

generated quantities {
  real lambda;   // phi[T-1] and p[T] not identified, but product is
  vector[M] LL;  // log-likelihood of each capture history
  
  lambda = phi[T-1] * p[T];
  
  // Likelihood of capture history, marginalized over discrete latent states
  LL = rep_vector(0,M);
  for (m in 1:M) 
  {
    if (last_minus_first[m] > 0)  // if history m was recaptured
    {
      for(t in (first[m]+1):last[m])
      {
        LL[m] += n[m] * log(phi[t-1]);                 // survival from t - 1 to t
        LL[m] += n[m] * bernoulli_lpmf(y[m,t] | p[t]); // observation (captured or not)
      }
    }
    LL[m] += n[m] * log(chi[last[m]]); // Pr[not detected after last[m]]
  }
}
