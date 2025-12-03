clear; clc; close all;

%% ------------------------------------------------------------------
% (A) 2-RC 참 파라미터 정의  [R0, R1, R2, tau1, tau2]
% ------------------------------------------------------------------
X_true = [0.001 0.0005 0.0005 6 60];

%% ------------------------------------------------------------------
% (B) 전류 프로파일 목록 (struct 또는 .xlsx 혼용 가능)
% ------------------------------------------------------------------
dt = 1; t_start = 0; t_end = 180; pulse_start = 0; pulse_end = 180;
pulse.t = (t_start:dt:t_end)';  pulse.I = zeros(size(pulse.t));
pulse.I(pulse.t>=pulse_start & pulse.t<=pulse_end) = 1;

driving_paths = {
    'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\udds_unit_time_scaled_current.xlsx'
    'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\us06_unit_time_scaled_current.xlsx'
    'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_CITY1_time_scaled_current.xlsx'
    'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_HW1_time_scaled_current.xlsx'
};

driving_files = driving_paths;

%% ------------------------------------------------------------------
% (C) 마르코프 노이즈 설정
% ------------------------------------------------------------------
epsilon_percent_span = 2;   
initial_state        = 51;
sigma                = 5;
nSeeds               = 10;

%% ------------------------------------------------------------------
% (D) MultiStart / fmincon 설정
% ------------------------------------------------------------------
ms        = MultiStart("UseParallel",true,"Display","off");
nStartPts = 20;
startPts  = RandomStartPointSet('NumStartPoints',nStartPts);
opt   = optimset('display','off','MaxIter',1e3,'MaxFunEvals',1e4,...
                 'TolFun',1e-14,'TolX',1e-15);

para0 = [0.0012 0.0006 0.0004 5 70]; 
lb    = [0 0 0 0.001 0.001];
ub    = para0*10;
A_lin = [0 0 0 1 -1]; b_lin = 0;

%% ------------------------------------------------------------------
% (E-0) 데이터 길이 절단 옵션
% ------------------------------------------------------------------
enable_cut        = true;
cut_lengths_fixed = [600, 300, 180];
min_cut_sec       = 180;

%% ------------------------------------------------------------------
% (E) 루프 시작
% ------------------------------------------------------------------
all_para_hats    = struct;
all_rmse         = struct;
all_data_size    = struct;   % 길이별 통계 저장
all_summary_size = struct;   % 엑셀/ MAT 저장용

for fileIdx = 1:numel(driving_files)

    % ===== 프로파일 로드 =====
    item = driving_files{fileIdx};
    if isstruct(item)
        t_full = item.t(:); I_full = item.I(:);
        base_name = sprintf('load%d',fileIdx);
    else
        tbl = readtable(item);
        t_full = tbl{:,1}; I_full = tbl{:,2};
        [~, base_name, ~] = fileparts(item);
    end
    OCV_full = zeros(size(t_full));
    T_full   = floor(t_full(end) - t_full(1));

    % ===== 사용할 길이 리스트 =====
    if enable_cut
        cut_list = unique([T_full, cut_lengths_fixed]);
        cut_list = cut_list(cut_list <= T_full & cut_list >= min_cut_sec);
    else
        cut_list = T_full;
    end
    cut_list = sort(cut_list, 'descend');
    tag_cut = ternary(enable_cut,'cutON','cutOFF');
    fprintf('\n==== [%s] length list (%s) : ', base_name, tag_cut);
    fprintf('%d ', cut_list); fprintf('\n');

    for ci = 1:numel(cut_list)
        T_keep = cut_list(ci);
        idx    = (t_full - t_full(1)) <= T_keep + 1e-9;
        t_vec  = t_full(idx);
        I_vec  = I_full(idx);
        OCV    = OCV_full(idx);

        %% 1) 참 전압 (노이즈 없음)
        V_est = RC_model_2(X_true,t_vec,I_vec,OCV);

        %% 2) 마르코프 노이즈
        noisedata = struct; noisedata.Non_noise = V_est;
        for seed = 1:nSeeds
            rng(seed);
            [noisy,~,~,~,~,~] = MarkovNoise_idx(V_est, epsilon_percent_span, initial_state, sigma);
            noisedata.(sprintf('V_SD%d',seed)) = noisy;
        end

        %% 3) MultiStart fitting (Non_noise + nSeeds)
        seednames = [{'Non_noise'}, arrayfun(@(s)sprintf('V_SD%d',s),1:nSeeds,'UniformOutput',false)];
        nSeries   = numel(seednames);
        para_hats = zeros(nSeries, numel(para0)+2);   % [R0 R1 R2 tau1 tau2 exitflag iter]
        RMSE_list = zeros(nSeries,1);

        figure('Name',sprintf('%s - Fit All Seeds (T=%ds)',base_name,T_keep), ...
               'NumberTitle','off');
        tiledlayout(3,4,'Padding','compact','TileSpacing','compact');

        for s = 1:nSeries
            V_SD = noisedata.(seednames{s});
            problem = createOptimProblem('fmincon', ...
                'objective',@(p) RMSE_2RC(V_SD,p,t_vec,I_vec,OCV), ...
                'x0',para0,'lb',lb,'ub',ub,...
                'Aineq',A_lin,'bineq',b_lin,'options',opt);
            [bestP,bestFval,eflg,~,sltns] = run(ms,problem,startPts);
            iter = NaN;
            if ~isempty(sltns)
                idxsol = find([sltns.Fval]==bestFval,1);
                iter   = sltns(idxsol).Output.iterations;
            end
            para_hats(s,:) = [bestP eflg iter];
            RMSE_list(s)   = bestFval;

            % (원하면 seed별 전압 플롯 넣을 수 있음)
        end
        sgtitle(sprintf('%s - 2RC Fitting (T=%ds, %s)',base_name,T_keep,tag_cut));

        %% 4) 통계값 (seed들 기준, Non_noise 제외)
        idxSeed   = 2:size(para_hats,1);
        mean_para = mean(para_hats(idxSeed,1:5),1);
        min_para  = min (para_hats(idxSeed,1:5),[],1);
        max_para  = max (para_hats(idxSeed,1:5),[],1);
        std_para  = std (para_hats(idxSeed,1:5),0,1);

        mean_RMSE = mean(RMSE_list(idxSeed));
        min_RMSE  = min (RMSE_list(idxSeed));
        max_RMSE  = max (RMSE_list(idxSeed));
        std_RMSE  = std (RMSE_list(idxSeed));

        fprintf('>> [%s | T=%4ds | %s] 파라미터 요약 통계 :\n', base_name, T_keep, tag_cut);
        fprintf('   [Mean] R0=%.8f, R1=%.8f, R2=%.8f, tau1=%.8f, tau2=%.8f, RMSE=%.8f\n', mean_para(1),mean_para(2),mean_para(3),mean_para(4),mean_para(5),mean_RMSE);
        fprintf('   [Min ] R0=%.8f, R1=%.8f, R2=%.8f, tau1=%.8f, tau2=%.8f, RMSE=%.8f\n', min_para(1), min_para(2), min_para(3), min_para(4), min_para(5), min_RMSE);
        fprintf('   [Max ] R0=%.8f, R1=%.8f, R2=%.8f, tau1=%.8f, tau2=%.8f, RMSE=%.8f\n', max_para(1), max_para(2), max_para(3), max_para(4), max_para(5), max_RMSE);
        fprintf('   [STD ] R0=%.8f, R1=%.8f, R2=%.8f, tau1=%.8f, tau2=%.8f, RMSE=%.8f\n\n', std_para(1), std_para(2), std_para(3), std_para(4), std_para(5), std_RMSE);

        %% 5) 결과 저장
        S = struct();
        S.size_sec = T_keep;
        S.mean = struct('R0',mean_para(1),'R1',mean_para(2),'R2',mean_para(3), ...
                        'tau1',mean_para(4),'tau2',mean_para(5),'RMSE',mean_RMSE);
        S.min  = struct('R0',min_para(1) ,'R1',min_para(2) ,'R2',min_para(3), ...
                        'tau1',min_para(4) ,'tau2',min_para(5) ,'RMSE',min_RMSE);
        S.max  = struct('R0',max_para(1) ,'R1',max_para(2) ,'R2',max_para(3), ...
                        'tau1',max_para(4) ,'tau2',max_para(5) ,'RMSE',max_RMSE);
        S.std  = struct('R0',std_para(1) ,'R1',std_para(2) ,'R2',std_para(3), ...
                        'tau1',std_para(4) ,'tau2',std_para(5) ,'RMSE',std_RMSE);

        all_data_size.(base_name).(sprintf('T_%ds',T_keep)) = S;
    end

    %% ------------------------------------------------------------------
    % (E-1) 이 프로파일에 대한 errorbar 플롯 (mean ± min/max)
    %% ------------------------------------------------------------------
    figure('Name',sprintf('%s - params vs length (mean±min/max) [%s]',base_name,tag_cut), ...
           'NumberTitle','off','Color','w');
    tiledlayout(3,2,'TileSpacing','compact','Padding','compact');

    fieldList  = {'R0','R1','R2','tau1','tau2','RMSE'};
    labels_tex = {'R_0','R_1','R_2','\tau_1','\tau_2','RMSE'};
    true_vals  = [X_true, NaN];   % RMSE는 True 없음

    names = fieldnames(all_data_size.(base_name));
    secs  = zeros(numel(names),1);
    for k = 1:numel(names)
        secs(k) = all_data_size.(base_name).(names{k}).size_sec;
    end

    T_full     = max(secs);
    want_order = unique([T_full, 600, 300, 180], 'stable');

    Ls = [];
    for L = want_order
        if any(secs==L), Ls(end+1) = L; end %#ok<AGROW>
    end
    nL = numel(Ls);

    for j = 1:6
        nexttile; hold on; grid on;

        meanVec = nan(nL,1);
        minVec  = nan(nL,1);
        maxVec  = nan(nL,1);

        for kk = 1:nL
            L   = Ls(kk);
            key = sprintf('T_%ds',L);
            Sj  = all_data_size.(base_name).(key);
            meanVec(kk) = Sj.mean.(fieldList{j});
            minVec(kk)  = Sj.min.(fieldList{j});
            maxVec(kk)  = Sj.max.(fieldList{j});
        end

        errLow  = meanVec - minVec;
        errHigh = maxVec  - meanVec;

        errorbar(Ls, meanVec, errLow, errHigh, '-o', ...
                 'LineWidth',1.5,'MarkerSize',6, ...
                 'DisplayName','Mean ± (Min/Max)');

        yTrue = true_vals(j);
        if ~isnan(yTrue)
            yline(yTrue,'--k','LineWidth',1.2,'DisplayName','True');
        end

        xlabel('Data length (s)');
        if j<=3
            ylabel(sprintf('%s (\\Omega)', labels_tex{j}));
        elseif j<=5
            ylabel(sprintf('%s (s)', labels_tex{j}));
        else
            ylabel(sprintf('%s (V)', labels_tex{j}));
        end
        title(sprintf('%s (mean±min/max)', labels_tex{j}), 'Interpreter','tex');

        % y축: True 값이 가운데 근처에 오도록
        if ~isnan(yTrue)
            yMinData = min(minVec, [], 'omitnan');
            yMaxData = max(maxVec, [], 'omitnan');
            spanLow  = yTrue - yMinData;
            spanHigh = yMaxData - yTrue;
            span     = max(spanLow, spanHigh);
            if ~isfinite(span) || span <= 0
                span = max(abs(yTrue), 1e-6);
            end
            yMin = yTrue - 1.1*span;
            yMax = yTrue + 1.1*span;
            if yMin < 0, yMin = 0; end
            ylim([yMin, yMax]);
        else
            yMin = min(minVec, [], 'omitnan');
            yMax = max(maxVec, [], 'omitnan');
            if ~isfinite(yMin) || yMin > 0, yMin = 0; end
            if ~isfinite(yMax) || yMax <= 0, yMax = 1; else, yMax = yMax*1.05; end
            ylim([yMin, yMax]);
        end

        if j==1
            legend('Location','best');
        end
    end

    %% ------------------------------------------------------------------
    % (E-2) 이 프로파일에 대한 요약 테이블 (Full/600/300/180 × Mean/Min/Max)
    %% ------------------------------------------------------------------
    names = fieldnames(all_data_size.(base_name));
    secs  = zeros(numel(names),1);
    for k=1:numel(names)
        secs(k) = all_data_size.(base_name).(names{k}).size_sec;
    end

    T_full     = max(secs);
    want_order = unique([T_full, 600, 300, 180], 'stable');

    have = [];
    for L = want_order
        if any(secs==L), have(end+1) = L; end %#ok<AGROW>
    end

    rowNames = {};
    for L = have
        rowNames = [rowNames; ...
            {sprintf('T%ds_Mean',L); sprintf('T%ds_Min',L); sprintf('T%ds_Max',L)}]; %#ok<AGROW>
    end

    Tprof = table( ...
        nan(numel(rowNames),1), nan(numel(rowNames),1), nan(numel(rowNames),1), ...
        nan(numel(rowNames),1), nan(numel(rowNames),1), nan(numel(rowNames),1), ...
        'VariableNames', {'R0','R1','R2','tau1','tau2','RMSE'}, ...
        'RowNames',     rowNames);

    r = 1;
    for L = have
        key = sprintf('T_%ds',L);
        S   = all_data_size.(base_name).(key);
        Tprof{r  ,:} = [S.mean.R0, S.mean.R1, S.mean.R2, S.mean.tau1, S.mean.tau2, S.mean.RMSE];
        Tprof{r+1,:} = [S.min.R0,  S.min.R1,  S.min.R2,  S.min.tau1,  S.min.tau2,  S.min.RMSE];
        Tprof{r+2,:} = [S.max.R0,  S.max.R1,  S.max.R2,  S.max.tau1,  S.max.tau2,  S.max.RMSE];
        r = r + 3;
    end

    all_summary_size.(base_name) = Tprof;
end

% %% === (F) 길이별 요약 저장 (MAT + Excel) ===========================
% save_root = fileparts(driving_paths{1});
% save_path = fullfile(save_root,'2RC_fitting');
% if ~exist(save_path,'dir'); mkdir(save_path); end
% 
% save(fullfile(save_path,'data_size_summary.mat'), 'all_summary_size', '-v7.3');
% 
% xlsx_path = fullfile(save_path, 'data_size_summary.xlsx');
% if exist(xlsx_path,'file'), delete(xlsx_path); end
% profiles = fieldnames(all_summary_size);
% for i = 1:numel(profiles)
%     Tprof = all_summary_size.(profiles{i});
%     writetable( ...
%         add_units_columns(Tprof), ...
%         xlsx_path, ...
%         'Sheet', profiles{i}, ...
%         'WriteRowNames', true ...
%     );
% end
% fprintf('✓ data_size_summary.mat / data_size_summary.xlsx 저장 완료: %s\n', save_path);

%% ------------------------------------------------------------------
% (Z) 모든 프로파일을 한 그림에: mean ± min/max + True (그림 1장)
% ------------------------------------------------------------------
profiles = fieldnames(all_data_size);
fieldList  = {'R0','R1','R2','tau1','tau2','RMSE'};
labels_tex = {'R_0','R_1','R_2','\tau_1','\tau_2','RMSE'};
true_vals  = [X_true, NaN];

colors = lines(numel(profiles));

figAll = figure('Name','[All Profiles] params vs data length (mean±min/max)', ...
                'NumberTitle','off','Color','w');
tiledlayout(3,2,'Padding','compact','TileSpacing','compact');

for j = 1:6
    nexttile; hold on; grid on;
    YminAll = inf; YmaxAll = -inf;

    for p = 1:numel(profiles)
        name  = profiles{p};
        nodes = fieldnames(all_data_size.(name));

        % 길이 및 mean/min/max 가져오기
        Ls  = zeros(numel(nodes),1);
        mV  = zeros(numel(nodes),1);
        mnV = zeros(numel(nodes),1);
        mxV = zeros(numel(nodes),1);

        for k = 1:numel(nodes)
            S   = all_data_size.(name).(nodes{k});
            Ls(k)  = S.size_sec;
            mV(k)  = S.mean.(fieldList{j});
            mnV(k) = S.min.(fieldList{j});
            mxV(k) = S.max.(fieldList{j});
        end

        [Ls,ord] = sort(Ls,'descend');
        mV  = mV(ord);
        mnV = mnV(ord);
        mxV = mxV(ord);

        errLow  = mV - mnV;
        errHigh = mxV - mV;

        errorbar(Ls, mV, errLow, errHigh, '-o', ...
                 'LineWidth',1.5,'MarkerSize',6, ...
                 'Color',colors(p,:), ...
                 'DisplayName',strrep(name,'_','\_'));

        YminAll = min(YminAll, min(mnV,[],'omitnan'));
        YmaxAll = max(YmaxAll, max(mxV,[],'omitnan'));
    end

    % True 값
    yTrue = true_vals(j);
    if ~isnan(yTrue)
        yline(yTrue,'--k','LineWidth',1.2,'DisplayName','True');
    end

    xlabel('Data length (s)');
    if j<=3
        ylabel(sprintf('%s (\\Omega)', labels_tex{j}));
    elseif j<=5
        ylabel(sprintf('%s (s)', labels_tex{j}));
    else
        ylabel(sprintf('%s (V)', labels_tex{j}));
    end
    title(sprintf('%s (mean±min/max)', labels_tex{j}), 'Interpreter','tex');

    % y축 스케일: True 값이 가운데쯤 (또는 없으면 데이터 기준)
    if ~isnan(yTrue)
        spanLow  = yTrue - YminAll;
        spanHigh = YmaxAll - yTrue;
        span     = max(spanLow, spanHigh);
        if ~isfinite(span) || span <= 0
            span = max(abs(yTrue), 1e-6);
        end
        yMin = yTrue - 1.1*span;
        yMax = yTrue + 1.1*span;
        if yMin < 0, yMin = 0; end
        ylim([yMin, yMax]);
    else
        if ~isfinite(YminAll) || YminAll > 0, YminAll = 0; end
        if ~isfinite(YmaxAll) || YmaxAll <= 0, YmaxAll = 1; else, YmaxAll = YmaxAll*1.05; end
        ylim([YminAll, YmaxAll]);
    end

    if j==1
        legend('Location','best');
    end
end

%% ------------------------------------------------------------------
% 보조 함수들
% ------------------------------------------------------------------
function cost = RMSE_2RC(data,para,t,I,OCV)
    model = RC_model_2(para,t,I,OCV);
    cost  = sqrt(mean((data - model).^2));
end

function out = ternary(cond,a,b)
    if cond, out=a; else, out=b; end
end

function Tout = add_units_columns(Tin)
% Tin: columns = R0 R1 R2 tau1 tau2 RMSE
% Units: Ω, Ω, Ω, s, s, V
u = table(repmat({'(Ω)'},height(Tin),1), repmat({'(Ω)'},height(Tin),1), repmat({'(Ω)'},height(Tin),1), ...
          repmat({'(s)'},height(Tin),1), repmat({'(s)'},height(Tin),1), repmat({'(V)'},height(Tin),1), ...
          'VariableNames', {'R0_Unit','R1_Unit','R2_Unit','tau1_Unit','tau2_Unit','RMSE_Unit'});
Tout = [Tin u];
end
