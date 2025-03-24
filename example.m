% Markov noise 모듈

function [noisy_I, states, final_state, P] = .MarkovNoise(I, epsilon_percent_span, initial_state)

    N = 101;  % 상태 개수
    epsilon_vector = linspace(-epsilon_percent_span/2, epsilon_percent_span/2, N);
    sigma = sigma_percent;

    P = zeros(N);
    for i = 1:N
        probabilities = normpdf(epsilon_vector, epsilon_vector(i), sigma);
        P(i, :) = probabilities / sum(probabilities);
    end

    current_state = initial_state;
    noisy_I = zeros(size(I));
    states  = zeros(size(I));

    for k = 1:length(I)
        
        eps_k = epsilon_vector(current_state);

        noisy_I(k) = I(k) + abs(I(k)) * eps_k;

        states(k) = current_state;

        current_state = randsample(1:N, 1, true, P(current_state, :));
    end

    final_state = states(end);

end
