% bimodal function

function [theta, tau, R1, R2, R_tot] = Bimodal_R(n, tau_min, tau_max, mu1, sigma1, mu2, sigma2, mode1, mode2)
%   Computes a bimodal resistance distribution R(τ) on an n-point log
%   grid of τ ∈ [tau_min, tau_max].  Two Gaussian modes in θ = ln(τ) are
%   centered at μ1, μ2 with standard deviations σ1, σ2.  Each mode’s total
%   area (∑ R·Δθ) is scaled to mode1 (Ω) and mode2 (Ω), respectively.
%
%   Inputs:
%     n        – number of τ points (positive integer)
%     tau_min  – minimum τ (positive scalar)
%     tau_max  – maximum τ (positive scalar > tau_min)
%     mu1      – θ-mean of mode 1  (i.e. ln(τ_peak1))
%     sigma1   – θ-std dev of mode 1
%     mu2      – θ-mean of mode 2  (i.e. ln(τ_peak2))
%     sigma2   – θ-std dev of mode 2
%     mode1    – total area for mode 1 (Ω), e.g. 0.001 for 1 mΩ
%     mode2    – total area for mode 2 (Ω), e.g. 0.001 for 1 mΩ
%
%   Outputs:
%     theta    – (n×1) vector of θ_i = ln(τ_i)
%     tau      – (n×1) vector of logarithmically spaced τ_i in [tau_min, tau_max]
%     R1       – (n×1) resistance contribution of mode 1 at each θ_i
%                (∑ R1_i · Δθ_i = mode1)
%     R2       – (n×1) resistance contribution of mode 2 at each θ_i
%                (∑ R2_i · Δθ_i = mode2)
%     R_tot    – (n×1) = R1 + R2.  Entire bimodal R distribution
%                (∑ R_tot_i · Δθ_i = mode1 + mode2)
%
%   Note: Δθ is computed by central‐difference on θ = ln(τ_i) to approximate
%   ∑[γ(θ_i)·Δθ_i] ≈ ∫γ(θ)dθ.  Each γ-mode is normalized so its ∑[γ·Δθ]=1,
%   then scaled by mode1 or mode2 in Ω.

%% 1) Create log‐spaced τ grid and corresponding θ
tau   = logspace(log10(tau_min), log10(tau_max), n).';  % (n×1)
theta = log(tau);                                       % θ_i = ln(τ_i)

%% 2) Compute un‐normalized γ1, γ2 at each θ_i
%    γ(θ) = (1/(σ sqrt(2π))) * exp(−(θ−μ)^2/(2σ^2))
gamma1 = (1./(sigma1 * sqrt(2*pi))) .* exp( - (theta - mu1).^2 ./ (2*sigma1^2) );
gamma2 = (1./(sigma2 * sqrt(2*pi))) .* exp( - (theta - mu2).^2 ./ (2*sigma2^2) );

%% 3) Compute Δθ_i via central‐difference
dtheta      = zeros(n,1);
dtheta(1)   = theta(2)   - theta(1);
dtheta(end) = theta(end) - theta(end-1);
for i = 2:n-1
    dtheta(i) = (theta(i+1) - theta(i-1)) / 2;
end

%% 4) Normalize each mode so ∑[γ·Δθ] = 1, then scale by mode1, mode2
%    ∑_{i=1}^n gamma1(i)*dtheta(i)  ≈ 1  after dividing
gamma1 = gamma1 ./ sum( gamma1 .* dtheta );  % Σ[γ1_i·Δθ_i] = 1
gamma2 = gamma2 ./ sum( gamma2 .* dtheta );  % Σ[γ2_i·Δθ_i] = 1

%    Now multiply by mode1, mode2 to get R1(i), R2(i):
R1    = gamma1 * mode1;  % Σ[R1_i·Δθ_i] = mode1 (Ω)
R2    = gamma2 * mode2;  % Σ[R2_i·Δθ_i] = mode2 (Ω)
R_tot = R1 + R2;         % Σ[R_tot_i·Δθ_i] = mode1 + mode2

end
