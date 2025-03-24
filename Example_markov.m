% Markov 사용예시

rng(0) % seed (고정됨. 0말고 다른 숫자 넣어도 됨. rng 없으면 랜덤으로 바뀜. 없어도 됨)

clc; clear; close all;

% (1) MarkovNoise 함수 호출
I = [0.5*ones(50,1); -1.0*ones(50,1)];  % 임의 전류 예시 (총 100개)
epsilon_percent_span = 4;  % 약 ±2%
initial_state = 50;        % 상태 인덱스(1~N 사이), 원하는 값 지정 (-> 함수에서 상태 개수 N=101(1~100) 로 설정. 50이 noise 0인 상태, 100은 위로, 1은 아래로 노이즈

[noisy_I, states, final_state, P] = MarkovNoise(I, epsilon_percent_span, initial_state);

% (2) 결과 확인
figure;
subplot(2,1,1);
plot(I, 'LineWidth',1.5); hold on;
plot(noisy_I, 'LineWidth',1.5);
xlabel('Sample Index'); ylabel('Current [A]');
legend('Original I','Noisy I');
title('Markov Noise Example');

subplot(2,1,2);
plot(states, 'LineWidth',1.5);
xlabel('Sample Index'); ylabel('State Index');
title('Markov Chain States');
