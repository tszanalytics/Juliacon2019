# Turing logistic example
"""
We need a logistic function, which is provided by StatsFuns.
"""
using StatsFuns: logistic, logit
using Turing, MCMCChains, BayesTesting

# Turing.jl model definitions:
# Bayesian logistic regression
# only one covariate (log time in the example below)
@model logistic_regression(x, y) = begin
    a ~ Normal(0, 5)
    b ~ Normal(0, 5)
#    balance ~ Normal(0, σ²)
#    income  ~ Normal(0, σ²)
    for i = 1:length(y)
        v = logistic(a + b*x[i])
        y[i] ~ Bernoulli(v)
    end
end

# multiple covariates in matrix x
@model logistic_multi_regression(x, y, ::Type{TV}=Vector{Float64}) where {TV} = begin
#    a ~ Normal(0, 10)
    n, k = size(x)
#    b = Array{Float64}(undef, k)
    b = TV(undef, k+1)
    for i in 1:(k+1)
        b[i] ~ Normal(0,6)
    end
    mu = [ones(n) x]*b
    for i = 1:n
        v = logistic(mu[i])
        y[i] ~ Bernoulli(v)
    end
end

# load data
using CSV, DataFrames, Plots, StatsPlots

df = CSV.read("cgi_example.csv")
# model CGI-I as a function of log(time)
y = df.cgi

x = df.ltime
n = length(y)

# Sample using HMC.
Random.seed!(1359)
@time chain = mapreduce(c -> sample(logistic_regression(x, y), HMC(3000, 0.05, 10)),
    chainscat,  1:3)

# Sample using NUTS
#chain = mapreduce(c -> sample(logistic_regression(x, y), NUTS(3000,1000, 0.65)),
#        chainscat,  1:3)

plot(chain)
cc = chain[1001:end]   ### must drop the adaption sample (1st 1000 here)
@show(describe(cc))
plot(cc)

bdraws = Array(cc["b"])
plot(bdraws, st=:density, label="b",fill=true)
adraws = Array(cc["a"])

## Generating the plots:

# plot predictive probabilities
# (apologies for the lazy cut & paste code, I should have wrote a loop!)
v = zeros(length(bdraws),5)
z = log.(1:5)
wk = [0 2 4 6 8]

## Corrected figure (the one in the presentation is incorrect)
plt = plot()
for i in 1:5
    v[:,i] = logistic.(adraws .+ bdraws.*z[i])
    if i == 1
        plot!(v[:,1], st=:density, label="Prob success for baseline",fill=true,title="Predictive Probabilities")
        vline!([mean(v[:,1])],linewidth=2,label="Baseline mean")
    else
        w = wk[i]
        plot!(v[:,i], st=:density, label="Prob success for x = $w",fill=true)
    end
end
plt
#savefig("predictive_probs.png")

yt = zeros(5)
ylb = zeros(5)
yub = zeros(5)
for i in 1:5
    yt[i] = mean(v[:,i])
    ylb[i],yub[i] = quantile(v[:,i],[0.025,0.975])
end

t = [0,2,4,6,8]
plot(t,yt, st=:scatter, color=:blue,label="Mean probability",legend=:topleft,xlabel="Weeks",ylabel="Probability")
plot!(t,ylb, st=:scatter, color=:green,alpha=0.6,label="0.95 interval")
plot!(t,yub, st=:scatter,color=:green,alpha=0.6, label="")
#savefig("trajectory_plot.png")

# An example with more RHS variables (covariates) - not in the presentation
# interaction variables to allow trajectory to differ
df.htr_ltime = df.ltime.*df.htr_over
df.c19_ltime = df.c19.*df.ltime
df.slc_ltime = df.slc_ss.*df.ltime
dd = dropmissing(df)  ## drop obs. with missing values
x = [dd.ltime dd.htr_over dd.htr_ltime]
y = dd.cgi

## The following takes approx 3 minutes on my machine for 5 chains
Random.seed!(1359)
nchains = 5  # number of MCMC chains
Turing.setadbackend(:reverse_diff)
# NUTS
# @time chain = mapreduce(c -> sample(logistic_multi_regression(x, y), NUTS(4000,2000, 0.65)),
#         chainscat,  1:nchains)
# HMC
@time chain = mapreduce(c -> sample(logistic_multi_regression(x, y), HMC(5000,0.05, 10)),
                chainscat,  1:nchains)

plot(chain)
cc = chain[1001:end]
plot(cc)
@show(describe(cc))

# plot each chain to select (if some unstable)
param = 2 # looking at the second parameter
plt = plot()
for i in 1:nchains
    c_param = Array(cc[:,param,i])
    plot!(c_param,st=:density,label="chain $i")
end
plt


b2_draws = Array(cc["b[2]"])  # get the b[2] draws
plot(b2_draws,st=:density,fill=true,label="b2")
vline!([0.0],label="",linecolor=:black,linewidth=2)
# This coefficient is not "statistically significant" - look where zero is
using BayesTesting
import BayesTesting.post_odds_pval
# Bayesian posterior density ratio and Bayesian probability in tail
@show(mcodds(b2_draws))
@show(bayespval(b2_draws))
pdr_pval(b2_draws)
####################################
