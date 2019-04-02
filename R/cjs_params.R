#' Compute survival and capture probabilities from a fitted multilevel CJS model
#'
#' @export
#' @param object Fitted model object of class `stanfit` representing a
#'   multilevel CJS model with `pars` `beta`, `epsilon`, `b`, and `e` (see XXX
#'   for definitions).
#' @param newX New `M x K` model matrix, as returned by `model.matrix`,
#'   containing exactly the same predictors, in the same order, as were passed
#'   as data variable `X` to `object`. Use the same `X` to calculate `phi` and
#'   `p` for the fitted data.
#' @param newgroups List with elements named `phi` and `p`, containing the
#'   matrices (`M x (T-1)` and `M x T`, respectively) of integer-valued group
#'   indices corresponding to `newX`. The unique indices for each occasion
#'   (column) must be a subset of those used for the corresponding occasion of
#'   `group_phi` or `group_p` passed to `object`. If there is only one group for
#'   a given parameter at a given occasion, group-varying random effects will
#'   not be used. Use the original `group_phi` and `group_p` to calculate `phi`
#'   and `p` for the fitted data.
#' @param N_mcmc Number of posterior draws to return. If `NULL` (the default),
#'   all samples in `object` are used.
#' @return A list whose elements `phi` and `p` are arrays (`N_mcmc x M x (T-1)`
#'   and `N_mcmc x M x T`, respectively) containing the posterior draws.
#'   

cjs_params <- function(object, newX, newgroups, N_mcmc = NULL) {
  beta <- extract(object, pars = "beta")$beta
  # epsilon <- extract(object, pars = "epsilon")$epsilon
  b <- extract(object, pars = "b")$b
  # e <- extract(object, pars = "e")$e
  if(is.null(N_mcmc)) N_mcmc <- nrow(beta)
  T <- dim(b)[3]
  M <- nrow(newX)
  K <- ncol(newX)
  group_phi <- newgroups$phi
  group_p <- newgroups$p
  phi <- array(dim = c(N_mcmc, M, T - 1))
  p <- array(dim = c(N_mcmc, M, T))
  
  for(t in 1:(T - 1))
    phi[,,t] <- plogis(tcrossprod(beta[,,t], newX))
  # phi[,,t] <- plogis(tcrossprod(beta[,,t], newX) + epsilon[group_phi[,t],t])
  
  for(t in 1:T)
    p[,,t] <- plogis(tcrossprod(b[,,t], newX))
  # p[,,t] <- plogis(tcrossprod(b[,,t], newX) + e[group_phi[,t],t])
  
  return(list(phi = phi, p = p))
}