% ======================================================================
%  SIM 기반 2-RC 피팅 + SOC70 구간 데이터 길이별 파라미터 요약 & 플롯
%
%  • 입력: *_SIM.mat (SIM_table 필요, 총 32 seg = 8×4 가정)
%  • 그룹핑: 기본=앞에서부터 8개씩 [90,70,50,30], 보조=SOC_center 최근접
%
%  • 출력A(기존, 토글): SOC70 길이별 요약 (Full / 600 / 300 / 180 s, Mean/Min/Max)
%                 -> SOC70_length_summary.xlsx (셀당 1시트, 8부하 평균)
%                 + 3×2 서브플롯 (R0~RMSE vs 데이터 길이[s], 셀별 라인)
%                 -> SOC70_params_vs_length.fig / .png
%
%  • 출력B(항상):  SOC70 + US06/UDDS만 길이별 파라미터 (셀×부하별)
%                 -> SOC70_US06_UDDS_forPPT.xlsx
%                    (행: 원본/600/300/180,
%                     열: US06_R0~tau2,US06_RMSE, UDDS_R0~tau2,UDDS_RMSE)
%                 + 3×2 서브플롯 (US06=주황계열, UDDS=파랑계열)
%                 -> SOC70_US06_UDDS_params_vs_length.fig / .png
%
%  ※ ENABLE_OLD_FEATURES = false 이면:
%     - 2RC 피팅은 SOC70 + (US06/UDDS) 세그먼트만 수행 (나머지 세그먼트는 스킵)
%     - 8부하 평균용 summary/엑셀/플롯은 생성하지 않음
% ======================================================================
clc; clear; close all;

%% ── 경로 & 파일 리스트 ───────────────────────────────────────────────
% folder_SIM = 'G:\공유 드라이브\BSL_Data4\HNE_SOC_moving_cutoff_5_processed\SIM_parsed';
% folder_SIM = 'G:\공유 드라이브\BSL_Data4\HNE_Integrated_6_processed\Test4(order3)\SIM_parsed';
% folder_SIM = 'G:\공유 드라이브\BSL_Data4\HNE_fresh_integrated_7_Drivingprocessed\SIM_parsed';
folder_SIM = 'G:\공유 드라이브\BSL_Data4\HNE_agedcell_8_processed\SIM_parsed\셀정렬';

sim_files  = dir(fullfile(folder_SIM,"*_SIM.mat"));
if isempty(sim_files)
    error("SIM 파일을 찾지 못했습니다: %s", folder_SIM);
end

% 저장 경로는 상위(SIM_parsed) 폴더로 고정
save_root = fileparts(folder_SIM);          % -> ...\SIM_parsed
save_path = fullfile(save_root,'2RC_fitting');
if ~exist(save_path,'dir'); mkdir(save_path); end

%% === 데이터 길이 옵션 (SOC70에서만 사용) ============================
len_targets_sec = [600 300 180];   % 잘라 쓸 목표 길이 [s]
min_len_sec     = 180;             % 최소 허용 길이
len_names_order = {'Full','T600s','T300s','T180s'};  % 길이 이름(표/플롯 공통 사용)
len_labels_kor  = {'원본길이','600 s','300 s','180 s'};  % PPT용 표 행 이름

%% === 기존 기능(8부하 평균 요약+플롯) 토글 =====================
ENABLE_OLD_FEATURES = false;  % false면 8종 부하 평균 SOC70_length_summary / avg 플롯/엑셀 안 만듦

%% === 플롯에서 제외할 셀(시트) 이름 설정 ==============================
%  -> base_field 기준 (matlab.lang.makeValidName 결과)
%exclude_cells_for_plot = {'HNE_fresh_8_1', 'HNE_fresh_22_3'}; 지우지마
exclude_cells_for_plot = {};

%% === (옵션) 셀 용량(QC40 등) 정보: 색 농도에 반영 ====================
USE_CAPACITY_FOR_COLOR = true;

% sim_files 순서와 동일한 순서로 용량 입력 (예: QC40 결과)
QC40_user = [57.49
             57.57
             54
             52.22
             53.45
             51.28
             57.91
             56.51
             42.14
             57.27
             57.18
             43.92
             58.40];
% QC40_user = [];   % 실제 사용할 땐 위처럼 채우고, 안 쓸 땐 [] 유지

%% ── fmincon + MultiStart 설정 ────────────────────────────────────────
ms       = MultiStart("UseParallel",true,"Display","off");
startPts = RandomStartPointSet('NumStartPoints',30);
opt      = optimset('display','off','MaxIter',1e3,'MaxFunEvals',1e4, ...
                    'TolFun',1e-14,'TolX',1e-15);

% 2-RC 초기추정값 / 경계 / 선형제약 (τ1<τ2)
para0 = [0.003 0.0005 0.0005 10 100]; 
lb    = [0       0       0      0.01  0.01];
ub    = [0.05 0.005 0.03 100 5000];
A_lin = [0 0 0 1 -1];  b_lin = 0;

%% ── 누적 컨테이너 ────────────────────────────────────────────────────
all_para_hats         = struct;   % 각 파일: [nSeg × 8]
all_rmse              = struct;   % 각 파일: [nSeg × 1]
all_len_summary_SOC70 = struct;   % 각 파일: 길이별 요약 테이블 (12×6, 8부하 평균)
cell_full_len_sec     = struct;   % 각 셀의 Full 길이(초, SOC70 세그 평균)

% (NEW) SOC70 + US06/UDDS 길이별 파라미터 저장용
US06_len_by_cell = struct;        % .(cell).param [4×6], .len [1×4]
UDDS_len_by_cell = struct;

% 대표 SOC(그룹 코드 매핑용)
soc_targets  = [90 70 50 30];

%% ── 메인 루프 (모든 파일 처리) ───────────────────────────────────────
for f = 1:numel(sim_files)
    fprintf('\n=== [%d/%d] 파일 처리 시작: %s ===\n', ...
        f, numel(sim_files), sim_files(f).name);

    % 1) load SIM_table
    S = load(fullfile(folder_SIM,sim_files(f).name),"SIM_table");
    if ~isfield(S,"SIM_table")
        warning("SIM_table 없음: %s (건너뜀)", sim_files(f).name);
        continue
    end
    SIM_table = S.SIM_table;

    base_raw   = erase(sim_files(f).name,"_SIM.mat");
    base_field = matlab.lang.makeValidName(base_raw);

    % 2) 이번 실험: 각 파일에 SIM 총 32개(=8×4) 가정
    nSeg = height(SIM_table);
    if nSeg==0
        warning("No SIM rows: %s", base_raw);
        continue
    end

    %% ── 그룹 코드 할당 (1:90, 2:70, 3:50, 4:30) ───────────────────
    grp_code = zeros(nSeg,1);
    if nSeg >= 32
        blk = [1 8; 9 16; 17 24; 25 32];
        for g = 1:4
            ii = blk(g,1):min(blk(g,2), nSeg);
            grp_code(ii) = g;
        end
    end

    % 예외 처리: SOC_center 기준 최근접 SOC 타겟으로 재할당
    if any(grp_code==0)
        SOC_center = mean([SIM_table.SOC1 SIM_table.SOC2], 2, 'omitnan');
        miss = isnan(SOC_center);
        if any(miss) && ismember('SOC_vec', SIM_table.Properties.VariableNames)
            try
                SOC_center(miss) = cellfun(@(v) mean(v,'omitnan'), SIM_table.SOC_vec(miss));
            catch
            end
        end
        valid = ~isnan(SOC_center);
        if any(valid)
            [~, gmin] = min(abs(SOC_center(valid) - soc_targets), [], 2);
            vidx = find(valid);
            grp_code(vidx) = gmin;
        end
    end

    %% 3) 2RC 피팅 ────────────────────────────────────────────────────
    %  ENABLE_OLD_FEATURES=true: 모든 세그먼트 피팅
    %  ENABLE_OLD_FEATURES=false: SOC70 + (US06, UDDS) 세그먼트만 피팅
    para_hats = nan(nSeg, 5+3);   % [R0 R1 R2 tau1 tau2 RMSE exitflag iter]
    RMSE_list = nan(nSeg, 1);

    for s = 1:nSeg
        if ~ENABLE_OLD_FEATURES
            % SOC70이 아니면 스킵
            if grp_code(s) ~= 2, continue; end
            % load index (1~8): 1=US06, 2=UDDS, 3=HWFET...
            loadIdx_in_block = mod(s-1,8) + 1;
            if ~(loadIdx_in_block==1 || loadIdx_in_block==2)
                continue;  % US06/UDDS가 아니면 스킵
            end
        end

        try
            t = SIM_table.time{s};        % duration or double
            I = SIM_table.current{s};
            V = SIM_table.voltage{s};
            O = []; 
            if ismember('OCV_vec', SIM_table.Properties.VariableNames)
                O = SIM_table.OCV_vec{s};
            end

            problem = createOptimProblem('fmincon', ...
              'objective',@(p)RMSE_2RC(V,p,t,I,O), ...
              'x0',para0,'lb',lb,'ub',ub, ...
              'Aineq',A_lin,'bineq',b_lin,'options',opt);

            [Pbest, Fval, exitflg, ~, sol] = run(ms,problem,startPts);
            it = NaN;
            if ~isempty(sol)
                it = sol(find([sol.Fval]==Fval,1)).Output.iterations;
            end

            para_hats(s,:) = [Pbest, Fval, exitflg, it];
            RMSE_list(s)   = Fval;

        catch ME
            warning("(%s) seg %d 피팅 실패: %s", base_raw, s, ME.message);
        end
    end

    %% 4) SOC70 구간만 사용하여 길이별 요약/US06/UDDS 계산 ───────────
    mask70 = (grp_code == 2);
    idx70  = find(mask70);
    fprintf('   → SOC70 세그먼트 개수: %d\n', numel(idx70));

    if isempty(idx70)
        warning('(%s) SOC70 세그먼트가 없어 length별 통계를 건너뜁니다.', base_raw);
    else
        % (A) 8부하 평균 summary용 변수 (old feature에서만 사용)
        if ENABLE_OLD_FEATURES
            len_targets = [NaN, len_targets_sec];   % NaN = Full(원본)

            rowNames_len = {};
            for kk = 1:numel(len_names_order)
                rowNames_len = [rowNames_len; ...
                    {sprintf('%s_Mean',len_names_order{kk}); ...
                     sprintf('%s_Min' ,len_names_order{kk}); ...
                     sprintf('%s_Max' ,len_names_order{kk})}]; %#ok<AGROW>
            end

            Tlen = table( ...
                nan(numel(rowNames_len),1), ...
                nan(numel(rowNames_len),1), ...
                nan(numel(rowNames_len),1), ...
                nan(numel(rowNames_len),1), ...
                nan(numel(rowNames_len),1), ...
                nan(numel(rowNames_len),1), ...
                'VariableNames', {'R0','R1','R2','tau1','tau2','RMSE'}, ...
                'RowNames', rowNames_len);

            P_len         = cell(numel(len_names_order),1);
            full_sec_list = [];
        else
            len_targets = [NaN, len_targets_sec];  % 구조 맞춰만 둠
        end

        % (B) SOC70에서 US06/UDDS만 따로 저장용
        load_param = struct( ...
            'US06', nan(numel(len_names_order),6), ...
            'UDDS', nan(numel(len_names_order),6));
        load_len   = struct( ...
            'US06', NaN(1,numel(len_names_order)), ...
            'UDDS', NaN(1,numel(len_names_order)));

        for k = 1:numel(idx70)
            s = idx70(k);        % SOC70에서의 k번째 세그먼트

            % --- 부하 index (1~8): 1=US06, 2=UDDS, 3=HWFET... ---
            loadIdx_in_block = mod(s-1,8) + 1;

            % --- 1) Full(원본 길이) 파라미터/오차: stage 3 결과 재사용 ---
            P_full   = para_hats(s,1:5);      % [R0 R1 R2 tau1 tau2]
            RMSE_ful = RMSE_list(s);          % full length RMSE

            if ENABLE_OLD_FEATURES
                % 8부하 평균용 full 파라미터 저장
                P_len{1} = [P_len{1}; [P_full, RMSE_ful]];
            end

            % --- 2) 시간 벡터(sec) 계산 & Full 길이 ---
            t = SIM_table.time{s};
            I = SIM_table.current{s};
            V = SIM_table.voltage{s};
            O = [];
            if ismember('OCV_vec', SIM_table.Properties.VariableNames)
                O = SIM_table.OCV_vec{s};
            end

            if isduration(t)
                tsec = seconds(t - t(1));
            elseif isdatetime(t)
                tsec = seconds(t - t(1));
            else
                tsec = t - t(1);
            end
            T_full_sec = tsec(end);

            % US06 full ≈ 599 s 이면 600 s로 간주 (threshold 예: ±10 s)
            T_eff_sec = T_full_sec;
            if loadIdx_in_block == 1  % US06
                if abs(T_full_sec - 600) <= 10
                    T_eff_sec = 600;
                end
            end

            if ENABLE_OLD_FEATURES
                full_sec_list(end+1) = T_eff_sec; %#ok<AGROW>
            end

            % --- 3) SOC70에서 US06/UDDS라면 full 길이 결과 저장 ---
            if loadIdx_in_block == 1
                tag_load = 'US06';
            elseif loadIdx_in_block == 2
                tag_load = 'UDDS';
            else
                tag_load = '';
            end
            if ~isempty(tag_load)
                load_param.(tag_load)(1,:) = [P_full, RMSE_ful];
                load_len.(tag_load)(1)     = T_eff_sec;   % 599 s도 600 s로 기록
            end

            % --- 4) 타겟 길이(600/300/180s)에 대해, 충분히 긴 세그먼트만 피팅 ---
            for kk = 2:numel(len_names_order)     % 2:600, 3:300, 4:180
                L = len_targets(kk);              % 목표 길이[s]

                % old 기능 꺼져있으면 US06/UDDS 아닌 부하는 여기서 스킵
                if ~ENABLE_OLD_FEATURES && isempty(tag_load)
                    continue;
                end

                % 길이 부족이면 스킵 (US06은 T_eff_sec 기준으로 판단)
                if T_eff_sec < max(L, min_len_sec)
                    continue;
                end

                idx_keep = (tsec <= L + 1e-9);
                t_vec = t(idx_keep);
                I_vec = I(idx_keep);
                V_vec = V(idx_keep);
                O_vec = [];
                if ~isempty(O)
                    O_vec = O(idx_keep);
                end

                problem_cut = createOptimProblem('fmincon', ...
                    'objective',@(p) RMSE_2RC(V_vec,p,t_vec,I_vec,O_vec), ...
                    'x0',para0,'lb',lb,'ub',ub, ...
                    'Aineq',A_lin,'bineq',b_lin,'options',opt);

                try
                    [Pbest_cut, Fval_cut, ~, ~, ~] = run(ms,problem_cut,startPts);

                    % (A) 8부하 평균용
                    if ENABLE_OLD_FEATURES
                        P_len{kk} = [P_len{kk}; [Pbest_cut(:)', Fval_cut]];
                    end

                    % (B) US06/UDDS라면 길이별 결과도 저장
                    if ~isempty(tag_load)
                        load_param.(tag_load)(kk,:) = [Pbest_cut(:)', Fval_cut];
                        load_len.(tag_load)(kk)     = L;
                    end
                catch ME
                    warning('(%s) SOC70 seg %d, L=%ds 피팅 실패: %s', ...
                        base_raw, s, L, ME.message);
                end
            end
        end

        % (A) 8부하 평균 summary 저장 (old 기능 on 일 때만)
        if ENABLE_OLD_FEATURES
            if ~isempty(full_sec_list)
                full_len_sec = mean(full_sec_list,'omitnan');
            else
                full_len_sec = NaN;
            end
            cell_full_len_sec.(base_field) = full_len_sec;

            rlen = 1;
            for kk = 1:numel(len_names_order)
                Pmat = P_len{kk};
                if ~isempty(Pmat)
                    Tlen{rlen  ,:} = mean(Pmat,1,'omitnan');   % Mean
                    Tlen{rlen+1,:} = min (Pmat,[],1);          % Min
                    Tlen{rlen+2,:} = max (Pmat,[],1);          % Max
                end
                rlen = rlen + 3;
            end
            all_len_summary_SOC70.(base_field) = Tlen;

            fprintf('[%s] SOC70 length-summary 완료 (Full≈%.1fs, 600/300/180s), loads=%d\n', ...
                base_raw, full_len_sec, numel(idx70));
        else
            fprintf('[%s] SOC70 US06/UDDS length-only 처리 완료 (8부하 평균 summary는 생략)\n', ...
                base_raw);
        end

        % (B) SOC70 + US06/UDDS 길이별 결과 저장 (항상)
        US06_len_by_cell.(base_field) = struct( ...
            'param', load_param.US06, ...
            'len',   load_len.US06);
        UDDS_len_by_cell.(base_field) = struct( ...
            'param', load_param.UDDS, ...
            'len',   load_len.UDDS);
    end

    all_para_hats.(base_field) = para_hats;
    all_rmse.(base_field)      = RMSE_list;

end

fprintf("\n모든 파일 처리 완료!\n");

%% ── (A) 기존 기능: SOC70 length summary + avg 플롯 (토글) ──────────
paramNames  = {'R0','R1','R2','tau1','tau2','RMSE'};
paramTitles = {'R_0 (\Omega)', 'R_1 (\Omega)', 'R_2 (\Omega)', ...
               '\tau_1 (s)', '\tau_2 (s)', 'RMSE (V)'};

if ENABLE_OLD_FEATURES
    % ── SOC70 length summary 엑셀 저장 (8부하 평균) ────────────────
    xlsx_len = fullfile(save_path,'SOC70_length_summary.xlsx');
    if exist(xlsx_len,'file'), delete(xlsx_len); end

    cells_len = fieldnames(all_len_summary_SOC70);
    for i = 1:numel(cells_len)
        Tlen = all_len_summary_SOC70.(cells_len{i});
        writetable(Tlen, xlsx_len, ...
            'Sheet', cells_len{i}, ...
            'WriteRowNames', true);
    end
    fprintf('SOC70 length summary 엑셀 저장 완료: %s\n', xlsx_len);

    % ── 3×2 서브플롯: 8부하 평균 길이별 Mean 값 (x축=초, 셀별 라인) ──
    cells_plot = cells_len;
    if ~isempty(exclude_cells_for_plot)
        mask_keep = ~ismember(cells_plot, exclude_cells_for_plot);
        cells_plot = cells_plot(mask_keep);
    end

    fig = figure('Name','SOC70 parameters vs data length (8 loads avg)', ...
                 'NumberTitle','off','Color','w', ...
                 'Position',[100 100 1500 800]);
    tl  = tiledlayout(3,2,'TileSpacing','compact','Padding','compact');

    for p = 1:numel(paramNames)
        pname = paramNames{p};
        ptit  = paramTitles{p};

        ax = nexttile; hold(ax,'on'); grid(ax,'on');

        Y_all = [];   % y값 범위 계산용
        X_all = [];   % x값(초) 범위 계산용

        for i = 1:numel(cells_plot)
            key  = cells_plot{i};
            Tlen = all_len_summary_SOC70.(key);

            if ~isfield(cell_full_len_sec, key) || isnan(cell_full_len_sec.(key))
                continue;
            end
            full_sec = cell_full_len_sec.(key);

            len_sec_all = [full_sec, 600, 300, 180];

            x = [];
            y = [];
            for kk = 1:numel(len_names_order)
                rname = sprintf('%s_Mean',len_names_order{kk});
                if ismember(rname, Tlen.Properties.RowNames)
                    val = Tlen{rname, pname};
                    if ~isnan(val)
                        x(end+1) = len_sec_all(kk); %#ok<AGROW>
                        y(end+1) = val;            %#ok<AGROW>
                    end
                end
            end

            if isempty(x), continue; end

            Y_all = [Y_all; y(:)]; %#ok<AGROW>
            X_all = [X_all; x(:)]; %#ok<AGROW>

            dname = strrep(key, '_', '\_');
            plot(ax, x, y, '-o', ...
                'LineWidth', 1.5, ...
                'MarkerSize', 6, ...
                'DisplayName', dname);
        end

        xlabel(ax,'Data length (s)');
        ylabel(ax, ptit, 'Interpreter','tex');
        title(ax, ptit, 'Interpreter','tex');
        legend(ax,'Location','best','Interpreter','tex');

        if ~isempty(Y_all)
            ymax = max(Y_all(:), [], 'omitnan');
            ymin = min(Y_all(:), [], 'omitnan');
            if ~isfinite(ymin) || ymin >= 0, ymin = 0; end
            if ~isfinite(ymax), ymax = 1; else, ymax = ymax*1.05; end
            ylim(ax,[ymin, ymax]);
        end

        if ~isempty(X_all)
            xmax = max(X_all);
            xlim(ax,[0, xmax*1.05]);
        end
    end

    title(tl,'SOC70 – 2RC parameters & RMSE vs data length (8 loads avg)','Interpreter','none');

    fig_name = 'SOC70_params_vs_length';
    savefig(fig, fullfile(save_path, [fig_name '.fig']));
    exportgraphics(fig, fullfile(save_path, [fig_name '.png']), 'Resolution', 200);
    fprintf('→ SOC70 3×2 subplot figure 저장 완료: %s\n', ...
        fullfile(save_path, [fig_name '.png']));
end

%% ── (B-1) NEW: SOC70 + US06/UDDS용 엑셀 (PPT 표 바로 복붙용, RMSE 포함) ─
xlsx_ppt   = fullfile(save_path,'SOC70_US06_UDDS_forPPT.xlsx');
if exist(xlsx_ppt,'file'), delete(xlsx_ppt); end

cells_ppt  = fieldnames(US06_len_by_cell);
usedSheets = {};   % 시트 이름 중복 방지용

% length 정보 누적용
len_cellNames = {};
len_US06_full = [];
len_UDDS_full = [];

for i = 1:numel(cells_ppt)
    key = cells_ppt{i};
    Su  = US06_len_by_cell.(key);
    Sd  = UDDS_len_by_cell.(key);

    US06_param = Su.param;  % [4×6] (R0 R1 R2 tau1 tau2 RMSE)
    UDDS_param = Sd.param;

    % 행: Full / 600 / 300 / 180 순서
    nRow = numel(len_labels_kor);
    US06_R0   = nan(nRow,1); US06_R1   = nan(nRow,1); US06_R2   = nan(nRow,1);
    US06_t1   = nan(nRow,1); US06_t2   = nan(nRow,1);
    US06_RMSE = nan(nRow,1);
    UDDS_R0   = nan(nRow,1); UDDS_R1   = nan(nRow,1); UDDS_R2   = nan(nRow,1);
    UDDS_t1   = nan(nRow,1); UDDS_t2   = nan(nRow,1);
    UDDS_RMSE = nan(nRow,1);

    for r = 1:nRow
        if r <= size(US06_param,1)
            US06_R0(r)   = US06_param(r,1) * 1e3;  % mΩ
            US06_R1(r)   = US06_param(r,2) * 1e3;
            US06_R2(r)   = US06_param(r,3) * 1e3;
            US06_t1(r)   = US06_param(r,4);
            US06_t2(r)   = US06_param(r,5);
            US06_RMSE(r) = US06_param(r,6);        % V
        end
        if r <= size(UDDS_param,1)
            UDDS_R0(r)   = UDDS_param(r,1) * 1e3;
            UDDS_R1(r)   = UDDS_param(r,2) * 1e3;
            UDDS_R2(r)   = UDDS_param(r,3) * 1e3;
            UDDS_t1(r)   = UDDS_param(r,4);
            UDDS_t2(r)   = UDDS_param(r,5);
            UDDS_RMSE(r) = UDDS_param(r,6);        % V
        end
    end

    Tppt = table( ...
        len_labels_kor', ...
        US06_R0, US06_R1, US06_R2, US06_t1, US06_t2, US06_RMSE, ...
        UDDS_R0, UDDS_R1, UDDS_R2, UDDS_t1, UDDS_t2, UDDS_RMSE, ...
        'VariableNames', { ...
            'Length_label', ...
            'US06_R0_mOhm','US06_R1_mOhm','US06_R2_mOhm','US06_tau1_s','US06_tau2_s','US06_RMSE_V', ...
            'UDDS_R0_mOhm','UDDS_R1_mOhm','UDDS_R2_mOhm','UDDS_tau1_s','UDDS_tau2_s','UDDS_RMSE_V'});

    % 엑셀 시트 이름 유효/유니크하게 변환
    sheetName = makeValidSheetName(key, usedSheets);
    usedSheets{end+1} = sheetName; %#ok<SAGROW>

    writetable(Tppt, xlsx_ppt, 'Sheet', sheetName, 'WriteRowNames', false);

    % 헤더용으로 full 길이도 출력 (PPT에서 "US06(XXX s)" 쓸 때 참고)
    full_US06 = Su.len(1);
    full_UDDS = Sd.len(1);
    fprintf('[%s | sheet:%s] US06 full ≈ %.0f s, UDDS full ≈ %.0f s (PPT 헤더에 사용)\n', ...
        key, sheetName, full_US06, full_UDDS);

    % (NEW) length 정보 누적
    len_cellNames{end+1,1} = sheetName;   % 시트 이름 기준으로 맞추기
    len_US06_full(end+1,1) = full_US06;
    len_UDDS_full(end+1,1) = full_UDDS;
end
fprintf('US06/UDDS용 PPT 엑셀(RMSE 포함) 저장 완료: %s\n', xlsx_ppt);

% (NEW) length 정보 시트 추가 (Cell별 US06/UDDS full length[s])
TlenInfo = table( ...
    len_cellNames, len_US06_full, len_UDDS_full, ...
    'VariableNames', {'Cell','US06_full_s','UDDS_full_s'});

writetable(TlenInfo, xlsx_ppt, 'Sheet','Length_sec','WriteRowNames',false);
fprintf('→ Length_sec 시트에 US06/UDDS full length[s] 저장 완료\n');

%% ── (B-2) NEW: SOC70 + US06/UDDS 3×2 플롯 (셀들 × 부하) ─────────────
cells_load  = fieldnames(US06_len_by_cell);
cells_plot2 = cells_load;
if ~isempty(exclude_cells_for_plot)
    mask_keep = ~ismember(cells_plot2, exclude_cells_for_plot);
    cells_plot2 = cells_plot2(mask_keep);
end

if isempty(cells_plot2)
    warning('US06/UDDS 플롯에 포함할 셀이 없습니다. exclude_cells_for_plot 확인.');
else
    nCells2 = numel(cells_plot2);

    baseBlue   = [0 0.4470 0.7410];      % UDDS
    baseOrange = [0.8500 0.3250 0.0980]; % US06
    color_UDDS = cell(nCells2,1);
    color_US06 = cell(nCells2,1);

    % 색 농도(alpha) 범위
    alpha_min = 0.35;
    alpha_max = 0.95;

    % --- 셀 용량(QC40_user) 기반 alpha 계산 시도 -------------------
    cap_vals = nan(nCells2,1);
    if USE_CAPACITY_FOR_COLOR && ~isempty(QC40_user) ...
            && numel(QC40_user)==numel(cells_load)
        % cells_load 순서와 QC40_user 순서가 동일하다고 가정
        for i = 1:nCells2
            % cells_plot2는 cells_load의 부분집합이므로 index 찾기
            idx_global = find(strcmp(cells_plot2{i}, cells_load), 1);
            if ~isempty(idx_global)
                cap_vals(i) = QC40_user(idx_global);
            end
        end
    end
    validCapIdx = find(~isnan(cap_vals));

    if USE_CAPACITY_FOR_COLOR && ~isempty(validCapIdx)
        % 유효한 용량들이 있을 때: 큰 용량일수록 alpha(진하게) ↑
        cap_min = min(cap_vals(validCapIdx));
        cap_max = max(cap_vals(validCapIdx));

        for i = 1:nCells2
            if isnan(cap_vals(i)) || cap_max==cap_min
                % 용량 정보 없거나 모두 같은 경우: 중간값
                alpha = 0.5*(alpha_min + alpha_max);
            else
                alpha = alpha_min + (cap_vals(i)-cap_min) / (cap_max-cap_min) ...
                                   * (alpha_max-alpha_min);
            end
            color_UDDS{i} = (1-alpha)*[1 1 1] + alpha*baseBlue;
            color_US06{i} = (1-alpha)*[1 1 1] + alpha*baseOrange;
        end
    else
        % 용량 정보가 전혀 없거나 길이 안 맞으면: index 기반 그라디언트
        for i = 1:nCells2
            alpha = alpha_min + (alpha_max-alpha_min)*(i-1)/max(nCells2-1,1);
            color_UDDS{i} = (1-alpha)*[1 1 1] + alpha*baseBlue;
            color_US06{i} = (1-alpha)*[1 1 1] + alpha*baseOrange;
        end
    end

    fig2 = figure('Name','SOC70 – US06/UDDS params vs length (cells)', ...
                  'NumberTitle','off','Color','w', ...
                  'Position',[100 100 1500 800]);
    tl2 = tiledlayout(3,2,'TileSpacing','compact','Padding','compact');

    scaleFactor = [1, 1, 1, 1, 1, 1];  % 필요하면 [1e3 1e3 1e3 1 1 1] 등으로 수정

    % ▶ 서브플롯에 들어갈 파라미터 순서 (타일 순서)
    %   1행: R0   | RMSE
    %   2행: R1   | tau1
    %   3행: R2   | tau2
    paramOrder = {'R0','RMSE','R1','tau1','R2','tau2'};

    for p = 1:numel(paramOrder)
        pname = paramOrder{p};

        % 제목 문자열은 기존 paramNames/paramTitles에서 가져옴
        idxTitle = strcmp(paramNames, pname);
        ptit     = paramTitles{idxTitle};

        ax = nexttile; hold(ax,'on'); grid(ax,'on');

        Y_all = [];
        X_all = [];

        for i = 1:nCells2
            key      = cells_plot2{i};
            niceName = strrep(key,'_','\_');

            % ---- US06 ----
            Su    = US06_len_by_cell.(key);
            len_u = Su.len(:);
            par_u = Su.param(:, pnameIndex(pname));
            par_u = par_u(:);
            mask_u = ~isnan(len_u) & ~isnan(par_u);
            if any(mask_u)
                x_u = len_u(mask_u);
                y_u = par_u(mask_u) * scaleFactor(p);
                Y_all = [Y_all; y_u]; %#ok<AGROW>
                X_all = [X_all; x_u]; %#ok<AGROW>
                if p==1
                    plot(ax, x_u, y_u, '-o', 'LineWidth',1.6, ...
                        'MarkerSize',6, 'Color',color_US06{i}, ...
                        'DisplayName',[niceName ' (US06)']);
                else
                    plot(ax, x_u, y_u, '-o', 'LineWidth',1.6, ...
                        'MarkerSize',6, 'Color',color_US06{i}, ...
                        'HandleVisibility','off');
                end
            end

            % ---- UDDS ----
            Sd    = UDDS_len_by_cell.(key);
            len_d = Sd.len(:);
            par_d = Sd.param(:, pnameIndex(pname));
            par_d = par_d(:);
            mask_d = ~isnan(len_d) & ~isnan(par_d);
            if any(mask_d)
                x_d = len_d(mask_d);
                y_d = par_d(mask_d) * scaleFactor(p);
                Y_all = [Y_all; y_d]; %#ok<AGROW>
                X_all = [X_all; x_d]; %#ok<AGROW>
                if p==1
                    plot(ax, x_d, y_d, '-s', 'LineWidth',1.6, ...
                        'MarkerSize',6, 'Color',color_UDDS{i}, ...
                        'DisplayName',[niceName ' (UDDS)']);
                else
                    plot(ax, x_d, y_d, '-s', 'LineWidth',1.6, ...
                        'MarkerSize',6, 'Color',color_UDDS{i}, ...
                        'HandleVisibility','off');
                end
            end
        end

        xlabel(ax,'Data length (s)');
        ylabel(ax, ptit, 'Interpreter','tex');
        title(ax, ptit, 'Interpreter','tex');

        if p==1
            legend(ax,'Location','best','Interpreter','tex');
        end

        if ~isempty(Y_all)
            ymax = max(Y_all(:), [], 'omitnan');
            ymin = min(Y_all(:), [], 'omitnan');
            if ~isfinite(ymin) || ymin >= 0, ymin = 0; end
            if ~isfinite(ymax), ymax = 1; else, ymax = ymax*1.05; end
            ylim(ax,[ymin ymax]);
        end
        if ~isempty(X_all)
            xmax = max(X_all);
            xlim(ax,[0 xmax*1.05]);
        end
    end


    title(tl2,'SOC70 – US06 / UDDS 2RC parameters vs data length (legend = cells)','Interpreter','none');

    fig2_name = 'SOC70_US06_UDDS_params_vs_length';
    savefig(fig2, fullfile(save_path, [fig2_name '.fig']));
    exportgraphics(fig2, fullfile(save_path, [fig2_name '.png']), 'Resolution', 200);
    fprintf('→ SOC70 US06/UDDS 3×2 subplot figure 저장 완료: %s\n', ...
        fullfile(save_path, [fig2_name '.png']));
end

fprintf('=== 모든 SOC70 length vs parameter 플롯 생성 완료 (3×2, x축=초) ===\n');

%% ── 보조 함수 ─────────────────────────────────────────────────────────
function cost = RMSE_2RC(V_true, para, t, I, OCV)
    V_est = RC_model_2(para,t,I,OCV);    % 사용자 정의 함수가 경로에 있어야 함
    cost  = sqrt(mean((V_true - V_est).^2));
end

function idx = pnameIndex(pname)
% 'R0','R1','R2','tau1','tau2','RMSE' -> 1..6 매핑
    switch pname
        case 'R0',   idx = 1;
        case 'R1',   idx = 2;
        case 'R2',   idx = 3;
        case 'tau1', idx = 4;
        case 'tau2', idx = 5;
        case 'RMSE', idx = 6;
        otherwise,   error('Unknown parameter name: %s', pname);
    end
end

function sheetName = makeValidSheetName(baseName, usedSheets)
% Excel 시트 이름 규칙 맞추기:
%  - 길이 1~31자
%  - 금지문자 : \ / ? * [ ]
%  - usedSheets 리스트와 중복되지 않도록 뒤에 _1,_2,... 붙임

    if isstring(baseName)
        baseName = char(baseName);
    end
    if ~ischar(baseName)
        baseName = char(string(baseName));
    end

    % 금지문자 제거/치환
    baseName = regexprep(baseName, '[:\\\/\?\*\[\]]', '_');

    % 빈 문자열 방지
    if isempty(baseName)
        baseName = 'Sheet1';
    end

    % 31자 제한
    if length(baseName) > 31
        baseName = baseName(1:31);
    end

    sheetName = baseName;
    k = 1;
    while any(strcmp(sheetName, usedSheets))
        suf    = ['_' num2str(k)];
        maxLen = 31 - length(suf);
        if maxLen < 1
            sheetName = suf(2:end);   % 최악의 경우
        else
            sheetName = [baseName(1:min(maxLen,length(baseName))) suf];
        end
        k = k + 1;
    end
end
