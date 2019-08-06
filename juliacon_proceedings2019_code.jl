using KernelDensity, CSV, Turing, MCMCChains, Distributions, Plots, StatsPlots

"""
PDR odds and p-value computation functions from BayesTesting
(including here to reduce package dependency)
x = MC sample, h0 = null hypothesis value
usage: odds, p_val, pval_2 = post_odds_pval(x, h0 = -2.3)
"""
function post_odds_pval(x; h0 = 0.0)
    d = kde(x)
    p,ind = findmax(d.density)
    ph0 = ifelse(pdf(d,h0) == 0.0,0.0000001,pdf(d,h0))
    odds = pdf(d,d.x[ind])/ph0
    p_val = length(x[x .<= h0])
    p_val2 = length(x[x .>= h0])
    if p_val <= p_val2
        p_value = p_val/length(x)
    else
        p_value = p_val2/length(x)
    end
    p_value_2tail = 2*p_value
    return odds, p_value, p_value_2tail
end

# CAMS data for Treatment and Placebo
df1 = CSV.read("jcp1.csv")
df4 = CSV.read("jcp2.csv")
wk121 = df1.wk121
wk124 = df4.wk124

# number of posterior simulation draws
M = 100000
### Proceedings Figure 1 and Table 1:
plt1 = plot()
println("n    odds   prob<0     p-value")
for i = 6:6:66
    q = i*2      #### need 2:1 ratio of treated to PCBO
    m1 = mean(wk121[1:q]); sd1 = std(wk121[1:q])    # group 1 Treatment
    m4 = mean(wk124[1:i]); sd4 = std(wk124[1:i])    # group 4 Placebo
    # Distn. of means with draws from analytical posterior t-distributiion
    # Can use MCMC instead here from conditional posteriors,
    # equations (3) and (4).
    t1draw = m1 .+ (sd1/sqrt(q)).*rand(TDist(q-1),M)
    t4draw = m4 .+ (sd4/sqrt(i)).*rand(TDist(i-1),M)
    dift1t4 = t1draw - t4draw      # Treatment vs. Placebo
#    @show(round.(post_odds_pval(dift1t4),digits=4))
    od, p_val, pval_2 = post_odds_pval(dift1t4)
    odds = round(od,digits=2)
    pval2 = round(pval_2,digits=4)
    pval = round(p_val,digits=4)
    println(3*i,"   ",odds,"   ",pval,"    ",pval2)

    if i == 6
        plt1 = plot(dift1t4, st=:density, alpha=(i*0.02*3/2),linewidth=2, xlims=[-12.0,5.0],ylims=[0.0,0.4],label="n = $(3*i)", xlabel="PARS difference",legend=:topleft)
        vline!([0.0],linewidth=2,linecolor=:black,label="")
    elseif i < 66
        plot!(dift1t4, st=:density, alpha=(i*0.02*3/2), linewidth=2,label="n = $(3*i)")
    else
        plot!(dift1t4, st=:density, alpha=1.2, linewidth=2,linecolor=:black,label="n = $(3*i)")
    end
end
current()
#savefig("pars_difference_sequential.png")

### BHM Example ###
# Turing.jl BHM for binomial trials
# model specification
@model binomial_trials(s,n,::Type{TV}=Vector{Float64}) where {TV} = begin
	g = length(n)  # number of groups
	# hierarchical priors
	ω ~ Beta(2,3)
	K ~ Gamma(10,1/0.05)
	a = ω*K + 1.0
	b = (1.0 - ω)*K + 1.0
	# priors for each occurrence rate
	θ = TV(undef, g)
#	θ = TV(undef, g)
	for k in 1:g
		θ[k] ~ Beta(a,b)
	end
	# likelihood
	for i in 1:g
		s[i] ~ Binomial(n[i],θ[i])
	end
end

# data used in example for Figures 3-5

### INSERT DATA HERE
# Placebo:
sn_pcbo = [3 11; 4 37; 8 65; 10 76; 2 150]
# Treatment:
sn_treat = [6 11; 7 37; 17 63; 23 133; 11 159]
# model estimation
Random.seed!(4243)
Turing.setadbackend(:reverse_diff)
# treatment

# NUTS
# @time ct = mapreduce(c->sample(binomial_trials(sn_treat[:,1], sn_treat[:,2]),
        NUTS(5000,1000,0.65)),chainscat,  1:5)

# HMC
@time ct = mapreduce(c->sample(binomial_trials(sn_treat[:,1], sn_treat[:,2]),
		HMC(5000, 0.05, 10)),chainscat,  1:5)

cc = ct[1001:end]
plot(cc)
@show(describe(cc))

# placebo
## N.B.: set following to NUTS(5000,1000,0.65) for NUTS
cp = mapreduce(c->sample(binomial_trials(sn_pcbo[:,1], sn_pcbo[:,2]),
        HMC(5000,0.05,10)), chainscat,  1:5)

# Figure 3 example:
cc = cp[1001:end]
plot(cc)
@show(describe(cc))


# Figure 4 example:
# (yes, the following could go in a loop, but copy & paste was faster!)
w_draws = Array(cc["ω"])
th1_draws = Array(cc["θ[1]"])
th2_draws = Array(cc["θ[2]"])
th3_draws = Array(cc["θ[3]"])
th4_draws = Array(cc["θ[4]"])
th5_draws = Array(cc["θ[5]"])
var_name = "Activation"

plot(w_draws, st=:density,fill=(0,0.4,:red),alpha=0.4, title="$var_name Posteriors SSRI Treatment",label="w",xlabel="Risk",ylabel="Probability density",legend=:topright)
plot!(th1_draws, st=:density,fill=(0,0.4,:blue),alpha=0.4, label="theta 1")
plot!(th2_draws, st=:density,fill=(0,0.4,:purple),alpha=0.4, label="theta 2")
plot!(th3_draws, st=:density,fill=(0,0.4,:green),alpha=0.4, label="theta 3")
plot!(th4_draws, st=:density,fill=(0,0.4,:yellow),alpha=0.4, label="theta 4")
plot!(th5_draws, st=:density,fill=(0,0.4,:pink),alpha=0.4, label="theta 5")
# savefig("activ_post_ssri_trt_uniform.png")


# Figure 5 example:
wt_draws = Array(ct["ω"][1001:end])  # treatment w
wp_draws = Array(cp["ω"][1001:end])  # placebo w
diff = wt_draws - wp_draws           # difference
plot(diff,st=:density,fill=(0,0.4,:purple),alpha=0.4,label="Difference")
vline!([0.0],linewidth=2,linecolor=:black,label="")
# compute mean, SD, odds and tail prob.
[mean(diff) std(diff)]
quantile(diff,[0.025,0.5,0.975])
# [mcodds(diff, h0=0.0) bayespval(diff)]  # If using BayesTesting package
pdr, p_value, p_value_2t = post_odds_pval(diff)

# assuming complete exchangility
# Treatment
ss = sum(sn_treat[:,1])
nn = sum(sn_treat[:,2])
thex_draws = rand(Beta(ss+1,nn-ss+1),100000)

plot(wt_draws, st=:density,fill=(0,0.4,:red),alpha=0.4, title="Posteriors SSRI Treatment",label="w (BHM)",xlabel="Risk",ylabel="Probability density",legend=:topright)
plot!(thex_draws,st=:density,fill=(0,0.4,:blue),alpha=0.4,label="theta (exchangeable)")
@show(quantile(thex_draws,[0.025,0.5,0.975]))
@show(quantile(wt_draws,[0.025,0.5,0.975]))

# savefig("SSRI_treat_activation_BHMvsExch_uniform.png")

@show([mean(wt_draws) std(wt_draws)])
@show(quantile(wt_draws,[0.025,0.5,0.975]))
pdrw, p_valuew, pv2tw = post_odds_pval(wt_draws)
pdrth, p_valueth, pv2th = post_odds_pval(thex_draws)
@show(pdrw, p_valuew, pv2tw)
@show(pdrth, p_valueth, pv2th)
# If using BayesTesting package:
# @show([mcodds(w_draws, h0=0.0) bayespval(w_draws,h0=0.0)])
# @show([mcodds(thex_draws, h0=0.0) bayespval(thex_draws,h0=0.0)])
