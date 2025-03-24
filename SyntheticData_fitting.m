% 합성데이터_fitting
clear; clc; close all

%% load data
pulsedata = 'pulse_data.mat';
syntheticdata = 'noise.mat';
load(pulsedata)
load(syntheticdata)

t = t_vec; 
V_SD = noisy; % 실제data
I = I_vec;
dt = 0.1; %[sec]

%% 1RC_fitting
% initial guess
para0 = [1.1 0.8 1.2]; %(20% 내외로 에러줌)

% bound 
lb= [0 0 0.001];
ub = para0*10;

% fitting
options = optimset('display','iter','MaxIter',400,'MaxFunEvals',1e5,...
    'TolFun',1e-10,'TolX',1e-8,'FinDiffType','central');
    % 'display','iter': 최적화 진행 동안 반복마다 결과 출력
    % 'MaxIter',400: 반복 횟수 최대 400회
    % 'MaxFunEvals',1e5: 함수평가 횟수 최대 100,000회
    % 'TolFun',1e-10: cost함수 값 변화가 e-10 보다 작아지면 종료
    % 'TolX',1e-8: para값 변화가 e-8보다 작아지면 종료
    % 'FinDiffType','central': 수치 미분 계산시 central differential 사용(보다 정확한 기울기 제공해줌)

para_hat = fmincon(@(para)RMSE_1RC(V_SD,para,t,I,dt),para0,[],[],[],[],lb,ub,[],options);    
    % fmincon 함수는 cost함수(RMSE_1RC)를 최소화하는 파라미터(para)를 찾음
    % @(para)RMSE_1RC(V_SD,para,t,I,dt): 익명함수 = 최적화 대상함수

%% present result
V_0 = RC_model_1(para0, t, I, dt);
V_hat = RC_model_1(para_hat, t, I, dt);

figure;
plot(t,V_SD, '-k'); hold on
plot(t,V_est,'-g');
plot(t,V_hat,'-r');
plot(t,V_0,'-b');

legend({'synthetic data','before synthesis','model','initial'});
xlabel('Time (sec)'); ylabel('Voltage [V]');
title('Synthetic Data Fitting');

%% cost funtion; RMSE (weight 무시)
function cost = RMSE_1RC(data,para,t,I,dt)
    % this is a cost function to be minimized
    model = RC_model_1(para, t, I, dt);
    cost = sqrt(mean((data - model).^2)); % RMSE error (data = V_SD; model = 최적화한 para 적용한 V_est)
end

