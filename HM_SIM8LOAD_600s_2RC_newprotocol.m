% ======================================================================
%  (개정-600s) 전체 SIM 기반 2-RC 피팅 + (동적 SOC) 전역 통계
%  + (NEW) 600초 윈도우 트리밍 후 피팅
%  + (NEW) 주행부하 8종(US06~HW2) 각각에 대해
%          Tbl_<LOAD>_ECM / Tbl_<LOAD>_RMSE 생성 (US06 방식과 동일)
%
%  가정(사용자 제공):
%   • 부하 순서: US06, UDDS, HWFET, WLTP, CITY1, CITY2, HW1, HW2 (총 8)
%   • blkSize=8 이면 각 SOC 블록 안에서 위 부하 순서대로 SIM이 진행
% ======================================================================

clc; clear; close all;

%% ── 경로 & 파일 리스트 ───────────────────────────────────────────────
folder_SIM = 'G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\20degC\이름정렬';

sim_files  = dir(fullfile(folder_SIM,"*_SIM.mat"));
if isempty(sim_files)
    error("SIM 파일을 찾지 못했습니다: %s", folder_SIM);
end

% 저장 경로는 상위(SIM_parsed) 폴더로 고정
save_root = fileparts(folder_SIM);          % -> ...\SIM_parsed

fit_window_sec = 600;  % ★ 핵심: 600초로 트리밍하여 피팅
save_path = fullfile(save_root, sprintf('2RC_fitting_%ds', fit_window_sec));
if ~exist(save_path,'dir'); mkdir(save_path); end

%% ── 사용자 설정: 존재 SOC 목록(여기만 바꾸면 됨) ─────────────────────
SOC_list = [50 70];          % 예: SOC 70, 50만 존재
nSOC     = numel(SOC_list);

% (옵션) 블록 그룹핑 사용 여부
use_block_mapping = true;
blkSize           = 8;        % ★ 부하 8종이면 blkSize=8 권장

%% ── 주행부하 이름/순서(고정) ─────────────────────────────────────────
loadNames = {'US06','UDDS','HWFET','WLTP','CITY1','CITY2','HW1','HW2'};
nLoads    = numel(loadNames);
if blkSize ~= nLoads
    warning('blkSize(%d) != nLoads(%d) 입니다. 블록 매핑이 의도대로 동작하지 않을 수 있습니다.', blkSize, nLoads);
end

%% ── RowNames 동적 생성: SOC70_Mean/Min/Max, SOC50_Mean/Min/Max ... ───
rowNames = strings(3*nSOC,1);
k = 0;
for si = 1:nSOC
    soc = SOC_list(si);
    rowNames(k+1) = sprintf('SOC%d_Mean', soc);
    rowNames(k+2) = sprintf('SOC%d_Min',  soc);
    rowNames(k+3) = sprintf('SOC%d_Max',  soc);
    k = k + 3;
end
rowNames = cellstr(rowNames);

%% ── fmincon + MultiStart 설정 ────────────────────────────────────────
ms       = MultiStart("UseParallel",true,"Display","off");
startPts = RandomStartPointSet('NumStartPoints',40);
opt      = optimset('display','off','MaxIter',1e3,'MaxFunEvals',1e4, ...
                    'TolFun',eps,'TolX',eps);

% 2-RC 초기추정값 / 경계 / 선형제약 (τ1<τ2)
para0 = [0.003 0.0005 0.0005 10 100];
lb    = [0       0       0      0.01  0.01];
ub    = [0.05 0.005 0.03 100 5000];
A_lin = [0 0 0 1 -1];  b_lin = 0;

%% ── 누적 컨테이너 ────────────────────────────────────────────────────
all_para_hats = struct;   % 각 파일: [nSeg × 8] = [R0 R1 R2 tau1 tau2 | RMSE exitflag iter] (R*는 Ω)
all_rmse      = struct;   % 각 파일: [nSeg × 1] RMSE (V)
all_summary   = struct;   % 각 파일: (3*nSOC)×6 요약 table (R0 R1 R2 tau1 tau2 RMSE; 각 SOC의 Mean/Min/Max)
                          %           R*는 mΩ, tau*는 s, RMSE는 V

all_load_idx  = struct;   % (NEW) 각 파일: [nSOC × nLoads] = SOC별/부하별 세그 인덱스

%% ── 메인 루프 (모든 파일 처리) ───────────────────────────────────────
for f = 1:numel(sim_files)

    % 1) load SIM_table
    S = load(fullfile(folder_SIM,sim_files(f).name),"SIM_table");
    if ~isfield(S,"SIM_table")
        warning("SIM_table 없음: %s (건너뜀)", sim_files(f).name);
        continue
    end
    SIM_table = S.SIM_table;

    base_raw   = erase(sim_files(f).name,"_SIM.mat");
    base_field = matlab.lang.makeValidName(base_raw);

    % 2) 세그 개수
    nSeg = height(SIM_table);
    if nSeg==0
        warning("No SIM rows: %s", base_raw);
        continue
    end

    % ── 그룹 코드 할당 (1..nSOC) ─────────────────────────────────────
    grp_code = zeros(nSeg,1);    % 1..nSOC, 0=미할당

    % (A) 블록 기반(옵션): nSeg가 충분하고 순서가 신뢰 가능할 때만
    if use_block_mapping && (nSeg >= blkSize*nSOC)
        for g = 1:nSOC
            ii = ((g-1)*blkSize + 1) : min(g*blkSize, nSeg);
            grp_code(ii) = g;
        end
    end

    % (B) 남은 미할당(또는 블록이 불가) → SOC_center 최근접 매핑
    if any(grp_code==0)
        SOC_center = mean([SIM_table.SOC1 SIM_table.SOC2], 2, 'omitnan');

        % SOC1/SOC2가 NaN이면 SOC_vec 평균으로 보강
        miss = isnan(SOC_center);
        if any(miss) && ismember('SOC_vec', SIM_table.Properties.VariableNames)
            try
                SOC_center(miss) = cellfun(@(v) mean(v,'omitnan'), SIM_table.SOC_vec(miss));
            catch
                % 그대로 NaN 유지
            end
        end

        valid = ~isnan(SOC_center);
        if any(valid)
            [~, gmin] = min(abs(SOC_center(valid) - SOC_list), [], 2);
            vidx = find(valid);
            grp_code(vidx(grp_code(vidx)==0)) = gmin(grp_code(vidx)==0);
        end
    end

    % ── (NEW) SOC×부하별 세그 인덱스 결정 ────────────────────────────
    load_idx = nan(nSOC, nLoads);
    for g = 1:nSOC
        for l = 1:nLoads
            load_idx(g,l) = pickLoadSegIdx(g, l, grp_code, nSeg, use_block_mapping, blkSize, nLoads);
        end
    end
    all_load_idx.(base_field) = load_idx;

    % 3) 전 세그먼트 피팅 (★ 600초 트리밍 적용)
    para_hats = nan(nSeg, 5+3);   % [R0 R1 R2 tau1 tau2 RMSE exitflag iter]
    RMSE_list = nan(nSeg, 1);     % RMSE (V)

    seg_list = 1:nSeg;

    % 서브플롯 레이아웃(최대 8열 고정, 행은 자동)
    cols  = 8;
    nPlot = numel(seg_list);
    rows  = max(1, ceil(nPlot/cols));

    fig = figure('Name',[base_raw sprintf(' – 2RC fitting (%ds)', fit_window_sec)], ...
        'NumberTitle','off', 'Position',[100 100 1600 900], 'Color','w');
    try
        sgtitle(strrep(base_raw,'_','\_') + sprintf(" – 2RC fitting (%ds)", fit_window_sec), 'Interpreter','tex');
    catch
        suptitle(strrep(base_raw,'_','\_') + sprintf(" – 2RC fitting (%ds)", fit_window_sec));
    end

    pcount = 0;
    for s = seg_list
        pcount = pcount + 1;

        try
            t = SIM_table.time{s};
            I = SIM_table.current{s};
            V = SIM_table.voltage{s};
            O = [];
            if ismember('OCV_vec', SIM_table.Properties.VariableNames)
                O = SIM_table.OCV_vec{s};
            end

            % 시간축 정규화
            if isduration(t)
                t = seconds(t - t(1));
            else
                t = t - t(1);
            end

            % ★ 600초 트리밍(필수)
            [t2, I2, V2, O2, okCrop] = cropToWindow(t, I, V, O, fit_window_sec);
            if ~okCrop
                warning("(%s) seg %d: 600s 트리밍 후 데이터가 부족하여 스킵", base_raw, s);
                continue;
            end

            problem = createOptimProblem('fmincon', ...
              'objective',@(p)RMSE_2RC(V2,p,t2,I2,O2), ...
              'x0',para0,'lb',lb,'ub',ub, ...
              'Aineq',A_lin,'bineq',b_lin,'options',opt);

            [Pbest, Fval, exitflg, ~, sol] = run(ms,problem,startPts);

            it = NaN;
            if ~isempty(sol)
                it = sol(find([sol.Fval]==Fval,1)).Output.iterations;
            end

            para_hats(s,:) = [Pbest, Fval, exitflg, it];  % R*는 Ω, tau*는 s
            RMSE_list(s)   = Fval;                        % V

            % ---- SOC/부하 라벨 ----
            soc_txt = 'SOC ?';
            if grp_code(s) >= 1 && grp_code(s) <= nSOC
                soc_txt = sprintf('SOC %d', SOC_list(grp_code(s)));
            end

            load_txt = 'LOAD ?';
            if grp_code(s) >= 1 && grp_code(s) <= nSOC
                g = grp_code(s);
                l = find(load_idx(g,:)==s, 1, 'first');
                if ~isempty(l)
                    load_txt = loadNames{l};
                end
            end

            % ---- 모델 전압(트리밍 구간 기준) ----
            V_fit = RC_model_2(Pbest, t2, I2, O2);
            V_ini = RC_model_2(para0 , t2, I2, O2);

            % ---- 서브플롯 ----
            subplot(rows, cols, pcount);
            plot(t2, V2, 'k', t2, V_fit, 'r', t2, V_ini, '--b', 'LineWidth', 1.1);
            grid on;
            xlabel('Time (s)'); ylabel('Voltage (V)');

            ttl = sprintf('SIM%d | %s | %s | RMSE=%.2f mV', s, load_txt, soc_txt, Fval*1e3);
            title(ttl, 'Interpreter','none');
            legend('True','Fitted','Initial','Location','northeast','Box','off');

        catch ME
            warning("(%s) seg %d 피팅 실패: %s", base_raw, s, ME.message);
        end
    end

    % (A) 파일별 피팅 figure 저장
    if isgraphics(fig,'figure')
       savefig(fig, fullfile(save_path, sprintf('%s_2RC_fit_%ds.fig', base_raw, fit_window_sec)));
    else
       warning('(%s) figure 핸들이 유효하지 않아 저장을 건너뜁니다.', base_raw);
    end

    % 4) SOC별 요약 테이블 구성: (3*nSOC)×6
    T = table( ...
        nan(3*nSOC,1), nan(3*nSOC,1), nan(3*nSOC,1), nan(3*nSOC,1), nan(3*nSOC,1), nan(3*nSOC,1), ...
        'VariableNames', {'R0','R1','R2','tau1','tau2','RMSE'}, ...
        'RowNames',     rowNames );

    P_all = para_hats(:,1:5);   % Ω, s

    groups = cell(nSOC,1);
    for g = 1:nSOC
        groups{g} = (grp_code==g);
    end

    mask_stat = true(nSeg,1);  % 여기서는 전체 seg를 SOC 통계에 사용(원래 방식 유지)

    r = 1;
    for g = 1:nSOC
        idx = groups{g} & mask_stat;
        if any(idx)
            blockP = P_all(idx,:);       % [*,5]
            blockE = RMSE_list(idx);     % [*,1]
            block6 = [blockP, blockE];   % [*,6]

            T{r  ,:} = mean(block6,1,'omitnan');  % Mean
            T{r+1,:} = min (block6,[],1);         % Min
            T{r+2,:} = max (block6,[],1);         % Max
        end
        r = r + 3;
    end

    % 저항 값을 mΩ 로 변환 (R0,R1,R2)
    T{:, {'R0','R1','R2'}} = 1000 * T{:, {'R0','R1','R2'}};

    % 5) 누적 저장
    all_para_hats.(base_field) = para_hats;  % R*는 Ω
    all_rmse.(base_field)      = RMSE_list;  % V
    all_summary.(base_field)   = T;          % R*는 mΩ

    % 로그
    cntStr = "";
    for g=1:nSOC
        cntStr = cntStr + sprintf(" SOC%d=%d", SOC_list(g), nnz(groups{g} & mask_stat));
    end
    fprintf('[done] %s → fitted %d segs (cropped %ds), summary(%d×6) 저장 | counts:%s\n', ...
        base_raw, numel(seg_list), fit_window_sec, 3*nSOC, cntStr);

end

fprintf("모든 파일 처리 완료!\n");

%% === ECM 평균 파라미터 테이블 (행: 셀, 열: SOC별 ECM Mean 파라미터) ===
cells_in_results = fieldnames(all_summary);
keys = strings(0,1);

for f = 1:numel(sim_files)
    base_raw   = erase(sim_files(f).name,"_SIM.mat");
    base_field = matlab.lang.makeValidName(base_raw);
    if ismember(base_field, cells_in_results)
        keys(end+1) = string(base_raw); %#ok<SAGROW>
    end
end

if isempty(keys)
    warning('ECM 테이블 생성 실패: all_summary 가 비어 있습니다.');
else
    cellRawNames = cellstr(keys);
    nCells       = numel(cellRawNames);

    pNames   = {'R0','R1','R2','tau1','tau2'};
    nP       = numel(pNames);

    ECM_mean_matrix = nan(nCells, numel(SOC_list)*nP);
    varNames        = strings(1, numel(SOC_list)*nP);

    col = 0;
    for si = 1:numel(SOC_list)
        soc = SOC_list(si);
        rowTag = sprintf('SOC%d_Mean', soc);

        for pi = 1:nP
            col = col + 1;
            p   = pNames{pi};

            if pi <= 3
                varNames(col) = sprintf('SOC%d_%s_mOhm', soc, p);
            else
                varNames(col) = sprintf('SOC%d_%s', soc, p);
            end

            for ci = 1:nCells
                key_field = matlab.lang.makeValidName(cellRawNames{ci});
                Tcell = all_summary.(key_field);
                ECM_mean_matrix(ci, col) = valOrNaN(Tcell, rowTag, p);
            end
        end
    end

    Tbl_ECM_mean = array2table(ECM_mean_matrix, ...
        'RowNames',      matlab.lang.makeValidName(cellRawNames), ...
        'VariableNames', cellstr(varNames));

    Tbl_ECM_mean.Properties.Description = ...
        sprintf('행: 셀(파일명) / 열: SOC별 ECM Mean 파라미터 (R* mΩ, tau s) | fit=%ds window', fit_window_sec);

    %% === SOC별 RMSE 요약 테이블 (단위: mV) ===========================
    statNames = ["avg","min","max"];
    nStat     = numel(statNames);

    RMSE_rowNames = strings(nSOC*nStat,1);
    for si = 1:nSOC
        soc     = SOC_list(si);
        rowBase = (si-1)*nStat;
        RMSE_rowNames(rowBase+1) = sprintf('SOC%d_avg', soc);
        RMSE_rowNames(rowBase+2) = sprintf('SOC%d_min', soc);
        RMSE_rowNames(rowBase+3) = sprintf('SOC%d_max', soc);
    end

    RMSE_matrix = nan(nSOC*nStat, nCells);

    for ci = 1:nCells
        key_field = matlab.lang.makeValidName(cellRawNames{ci});
        Tcell     = all_summary.(key_field);   % RMSE는 V 단위

        for si = 1:nSOC
            soc     = SOC_list(si);
            rowBase = (si-1)*nStat;

            tagMean = sprintf('SOC%d_Mean', soc);
            tagMin  = sprintf('SOC%d_Min',  soc);
            tagMax  = sprintf('SOC%d_Max',  soc);

            RMSE_matrix(rowBase+1, ci) = valOrNaN(Tcell, tagMean, 'RMSE') * 1e3;
            RMSE_matrix(rowBase+2, ci) = valOrNaN(Tcell, tagMin,  'RMSE') * 1e3;
            RMSE_matrix(rowBase+3, ci) = valOrNaN(Tcell, tagMax,  'RMSE') * 1e3;
        end
    end

    Tbl_RMSE = array2table(RMSE_matrix, ...
        'RowNames',      cellstr(RMSE_rowNames), ...
        'VariableNames', matlab.lang.makeValidName(cellRawNames));

    Tbl_RMSE.Properties.Description = ...
        sprintf('행: SOC별 RMSE(avg/min/max, mV) / 열: 셀(파일명) | fit=%ds window', fit_window_sec);

    %% === (NEW) 주행부하별 Tbl_<LOAD>_ECM, Tbl_<LOAD>_RMSE 생성 =========
    % - US06에서 하던 방식 그대로: 각 부하에서 SOC별로 "해당 seg 1개 값" 저장

    pNames2 = {'R0','R1','R2','tau1','tau2'};
    nP2     = numel(pNames2);

    % 결과를 구조체로도 묶어둠 (편의)
    Tbl_Load_ECM  = struct;
    Tbl_Load_RMSE = struct;

    for l = 1:nLoads
        loadName = loadNames{l};

        % ECM (행=셀, 열=SOC별 R0~tau2)
        ECM_mat  = nan(nCells, numel(SOC_list)*nP2);
        vnamesL  = strings(1, numel(SOC_list)*nP2);

        col = 0;
        for si = 1:numel(SOC_list)
            soc = SOC_list(si);

            for pi = 1:nP2
                col = col + 1;
                p   = pNames2{pi};

                if pi <= 3
                    vnamesL(col) = sprintf('SOC%d_%s_mOhm', soc, p);
                else
                    vnamesL(col) = sprintf('SOC%d_%s', soc, p);
                end

                for ci = 1:nCells
                    key_field   = matlab.lang.makeValidName(cellRawNames{ci});
                    load_idx_ci = all_load_idx.(key_field);   % [nSOC×nLoads]
                    sIdx = load_idx_ci(si, l);

                    if ~isnan(sIdx) && sIdx>=1
                        Prow = all_para_hats.(key_field);     % [nSeg×8], R*는 Ω
                        if sIdx <= size(Prow,1)
                            val = Prow(sIdx, pi);             % pi=1..5
                            if pi <= 3
                                val = val * 1000;             % Ω -> mΩ
                            end
                            ECM_mat(ci, col) = val;
                        end
                    end
                end
            end
        end

        T_ECM = array2table(ECM_mat, ...
            'RowNames',      matlab.lang.makeValidName(cellRawNames), ...
            'VariableNames', cellstr(vnamesL));

        T_ECM.Properties.Description = sprintf('행: 셀 / 열: SOC별 %s 2RC 파라미터 (R*mΩ, tau s) | fit=%ds', loadName, fit_window_sec);

        % RMSE (행=SOC, 열=셀, mV)
        RMSE_matL = nan(nSOC, nCells);
        rnamesL   = strings(nSOC,1);
        for si = 1:nSOC
            rnamesL(si) = sprintf('SOC%d', SOC_list(si));
        end

        for ci = 1:nCells
            key_field   = matlab.lang.makeValidName(cellRawNames{ci});
            load_idx_ci = all_load_idx.(key_field);

            for si = 1:nSOC
                sIdx = load_idx_ci(si, l);
                if ~isnan(sIdx) && sIdx>=1
                    Erow = all_rmse.(key_field);  % V
                    if sIdx <= numel(Erow)
                        RMSE_matL(si, ci) = Erow(sIdx) * 1e3; % mV
                    end
                end
            end
        end

        T_RMSE = array2table(RMSE_matL, ...
            'RowNames',      cellstr(rnamesL), ...
            'VariableNames', matlab.lang.makeValidName(cellRawNames));

        T_RMSE.Properties.Description = sprintf('행: SOC / 열: 셀 / 값: %s RMSE (mV) | fit=%ds', loadName, fit_window_sec);

        % 구조체 저장 + (옵션) 개별 변수도 생성
        Tbl_Load_ECM.(loadName)  = T_ECM;
        Tbl_Load_RMSE.(loadName) = T_RMSE;

        % 개별 변수 (US06 방식 그대로 쓰고 싶으면)
        assignin('base', sprintf('Tbl_%s_ECM',  loadName), T_ECM);
        assignin('base', sprintf('Tbl_%s_RMSE', loadName), T_RMSE);
    end
end

%% 결과 저장
results_file = fullfile(save_path, sprintf('2RC_results_%ds.mat', fit_window_sec));

vars_to_save = {'all_para_hats','all_rmse','all_summary','all_load_idx'};
if exist('Tbl_ECM_mean','var'), vars_to_save{end+1}='Tbl_ECM_mean'; end %#ok<AGROW>
if exist('Tbl_RMSE','var'),     vars_to_save{end+1}='Tbl_RMSE';     end %#ok<AGROW>
if exist('Tbl_Load_ECM','var'), vars_to_save{end+1}='Tbl_Load_ECM'; end %#ok<AGROW>
if exist('Tbl_Load_RMSE','var'),vars_to_save{end+1}='Tbl_Load_RMSE';end %#ok<AGROW>

save(results_file, vars_to_save{:}, '-v7.3');
fprintf('2RC 결과 저장 완료: %s\n', results_file);

%% ── 보조 함수 ─────────────────────────────────────────────────────────
function cost = RMSE_2RC(V_true, para, t, I, OCV)
    V_est = RC_model_2(para,t,I,OCV);
    cost  = sqrt(mean((V_true - V_est).^2, 'omitnan'));
end

function y = valOrNaN(T, rowName, colName)
    y = NaN;
    if ismember(rowName, T.Properties.RowNames)
        try
            y = T{rowName, colName};
            if isempty(y), y = NaN; end
        catch
            y = NaN;
        end
    end
end

function [t2, I2, V2, O2, ok] = cropToWindow(t, I, V, O, winSec)
    % t는 0부터 시작한다고 가정(메인에서 보장)
    ok = true;

    if isempty(t) || numel(t) < 5
        ok = false; t2=[]; I2=[]; V2=[]; O2=[];
        return
    end

    m = (t <= winSec);
    if nnz(m) < 5
        ok = false; t2=[]; I2=[]; V2=[]; O2=[];
        return
    end

    t2 = t(m);
    I2 = I(m);
    V2 = V(m);

    if isempty(O)
        O2 = [];
    else
        try
            O2 = O(m);
        catch
            O2 = [];
        end
    end
end

function sIdx = pickLoadSegIdx(g, loadIdx, grp_code, nSeg, use_block_mapping, blkSize, nLoads)
    % SOC그룹 g(1..nSOC)에서 부하(loadIdx:1..nLoads)에 해당하는 seg index를 선택
    % - 블록 매핑 가능: (g-1)*blkSize + loadIdx
    % - fallback: 해당 SOC 그룹의 seg들을 "등장 순서대로" 정렬해서 loadIdx번째를 사용
    %
    % 반환: sIdx (double), 실패 시 NaN

    sIdx = NaN;

    % 1) 블록 기반(권장)
    if use_block_mapping && (blkSize >= nLoads) && (nSeg >= blkSize*max(g,1))
        tmp = (g-1)*blkSize + loadIdx;
        if tmp >= 1 && tmp <= nSeg
            sIdx = tmp;
            return
        end
    end

    % 2) fallback: grp_code==g 중 loadIdx번째(등장 순서대로)
    idx = find(grp_code==g);
    if isempty(idx), return; end
    idx = sort(idx(:));

    if numel(idx) >= loadIdx
        sIdx = idx(loadIdx);
    end
end
