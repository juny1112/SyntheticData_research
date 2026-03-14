%% ======================================================================
%  ECM(tau1,tau2) vs PSD ALL features Correlation Analysis (FULL)
%
%  - ECM: load from 2RC_results_600s.mat (Tbl_Load_ECM)
%  - PSD: load psd_stat_tbl from saved .mat (or workspace)
%
%  What it does
%   1) Build ECM long table: (LoadKey × SOC) -> tau1_s, tau2_s
%   2) Load PSD table: (LoadKey) -> PSD features (mean/std/int/band energies/ratio...)
%   3) Join -> J (LoadKey × SOC) with all PSD features broadcast to SOC rows
%   4) For each SOC:
%        - compute corr between (tau1, tau2) and ALL numeric PSD features
%        - plot ONE figure per SOC with subplot grid: 2 rows (tau1,tau2) × nFeat cols
%   5) Save J and correlation tables
%
%  Options:
%   - use_log: false => linear corr/plot
%              true  => log-log corr/plot
%   - corr_type: "Pearson" (default) or "Spearman"
%
%  NOTE:
%   - numeric PSD features are auto-detected from psd_stat_tbl (excluding keys/fs/dt if desired)
%   - uses "Rows","complete" to ignore NaNs
%% ======================================================================
clear; clc; close all;

%% (0) INPUT --------------------------------------------------------------
ecm_results_file = "G:\공유 드라이브\BSL_Data4\HNE_fresh_integrated_7_Drivingprocessed\TS_2RC_fitting_600s\2RC_results_600s.mat";

use_psd_from_workspace = false;
psd_saved_file = "G:\공유 드라이브\Battery Software Lab\Protocols\2025 현대 NE 고품셀 평가 실험\주행부하_scaledown\PSD\psd_stat_tbl.mat";

SOC_list = [90 70 50 30];

save_dir = "G:\공유 드라이브\BSL_Data4\HNE_fresh_integrated_7_Drivingprocessed\TS_2RC_fitting_600s\PSD_vs_ECM";
if ~exist(save_dir,'dir'); mkdir(save_dir); end

use_outerjoin_debug = false;   % true면 누락 확인용 outerjoin
use_cell_mean = false;         % ECM에서 load별 셀 평균 쓸지
rowi = 1;                      % use_cell_mean=false일 때 사용할 row index

% ===== 분석 옵션 =====
use_log   = true;             % false=linear, true=log-log
corr_type = "Spearman";         % "Pearson" or "Spearman"

% PSD feature에서 제외하고 싶은 컬럼(필요시 추가)
excludePSD = ["Load","LoadKey","fs_Hz","dt_s"];  % 보통 키/샘플링정보는 feature에서 제외

fprintf("=== SETTINGS ===\n");
fprintf("use_log   = %d\n", use_log);
fprintf("corr_type = %s\n\n", corr_type);

%% (1) ECM LOAD -----------------------------------------------------------
S = load(ecm_results_file, "Tbl_Load_ECM");
if ~isfield(S,"Tbl_Load_ECM")
    error("Tbl_Load_ECM 이 ECM 결과 파일에 없습니다: %s", ecm_results_file);
end
Tbl_Load_ECM = S.Tbl_Load_ECM;

loadKeys_ECM_raw = string(fieldnames(Tbl_Load_ECM));  % raw struct fields
loadKeys_ECM     = normalizeLoadKey(loadKeys_ECM_raw);

%% (2) PSD LOAD -----------------------------------------------------------
if use_psd_from_workspace
    if ~exist("psd_stat_tbl","var")
        error("workspace에 psd_stat_tbl이 없습니다. PSD 스크립트를 먼저 실행하거나 psd_saved_file로 불러오세요.");
    end
    psd = psd_stat_tbl;
else
    if strlength(psd_saved_file)==0 || ~isfile(psd_saved_file)
        error("psd_saved_file이 비어있거나 파일이 없습니다: %s", psd_saved_file);
    end
    Sp = load(psd_saved_file, "psd_stat_tbl");
    if ~isfield(Sp,"psd_stat_tbl")
        error("psd_saved_file 안에 psd_stat_tbl 변수가 없습니다: %s", psd_saved_file);
    end
    psd = Sp.psd_stat_tbl;
end

% PSD key normalize
if ~ismember("Load", string(psd.Properties.VariableNames))
    error("psd_stat_tbl에 'Load' 컬럼이 없습니다. PSD 스크립트 출력을 확인하세요.");
end
psd.LoadKey = normalizeLoadKey(psd.Load);
psd = movevars(psd, "LoadKey", "Before", 1);

%% (2-1) KEY MATCH CHECK --------------------------------------------------
uPSD = unique(psd.LoadKey);
uECM = unique(loadKeys_ECM);

missing_in_PSD = setdiff(uECM, uPSD);
missing_in_ECM = setdiff(uPSD, uECM);

if ~isempty(missing_in_PSD)
    warning("ECM에 있는데 PSD에 없는 LoadKey: %s", strjoin(missing_in_PSD, ", "));
end
if ~isempty(missing_in_ECM)
    warning("PSD에 있는데 ECM에 없는 LoadKey: %s", strjoin(missing_in_ECM, ", "));
end

%% (3) ECM LONG TABLE: (LoadKey × SOC) -> tau1/tau2 -----------------------
ecm_long = table();

for li = 1:numel(loadKeys_ECM_raw)
    Lraw = loadKeys_ECM_raw(li);
    Lkey = normalizeLoadKey(Lraw);

    T = Tbl_Load_ECM.(Lraw);

    for soc = SOC_list
        v_tau1 = sprintf("SOC%d_tau1", soc);
        v_tau2 = sprintf("SOC%d_tau2", soc);

        if ~ismember(v_tau1, string(T.Properties.VariableNames)) || ~ismember(v_tau2, string(T.Properties.VariableNames))
            continue;
        end

        if use_cell_mean
            tau1 = mean(T{:, v_tau1}, 'omitnan');
            tau2 = mean(T{:, v_tau2}, 'omitnan');
        else
            if height(T) < rowi
                continue;
            end
            tau1 = T{rowi, v_tau1};
            tau2 = T{rowi, v_tau2};
        end

        ecm_long = [ecm_long;
            table(Lkey, soc, tau1, tau2, ...
            'VariableNames', {'LoadKey','SOC','tau1_s','tau2_s'})]; %#ok<AGROW>
    end
end

if isempty(ecm_long)
    error("ecm_long이 비었습니다. SOC_list 또는 Tbl_Load_ECM 변수명을 확인하세요.");
end

%% (4) JOIN ---------------------------------------------------------------
if use_outerjoin_debug
    J = outerjoin(ecm_long, psd, "Keys","LoadKey", "MergeKeys",true);
else
    J = innerjoin(ecm_long, psd, "Keys","LoadKey");
end

% Bring key vars forward
wantFirst = ["LoadKey","SOC","tau1_s","tau2_s"];
J = movevars(J, intersect(wantFirst, string(J.Properties.VariableNames), 'stable'), "Before", 1);

disp("==== Joined table preview (head) ====");
disp(head(J, 20));

%% (5) AUTO-DETECT PSD NUMERIC FEATURES ----------------------------------
varNames = string(J.Properties.VariableNames);

% PSD 후보: psd 테이블에서 온 변수들 중 numeric인 것만
psdVarsAll = string(psd.Properties.VariableNames);
psdVarsAll(psdVarsAll=="LoadKey") = []; % join key
% (주의) psd.Load는 string이라 numeric 아니지만, 혹시 제외목록에 포함
psdVarsUse = setdiff(psdVarsAll, excludePSD, 'stable');

% J 안에 존재하는 것만
psdVarsUse = intersect(psdVarsUse, varNames, 'stable');

% numeric만 추리기
isNum = false(size(psdVarsUse));
for i = 1:numel(psdVarsUse)
    isNum(i) = isnumeric(J.(psdVarsUse(i)));
end
psdNumVars = psdVarsUse(isNum);

if isempty(psdNumVars)
    error("PSD numeric feature를 찾지 못했습니다. excludePSD 설정/psd_stat_tbl 컬럼을 확인하세요.");
end

fprintf("=== PSD numeric features used (%d) ===\n", numel(psdNumVars));
fprintf("%s\n\n", strjoin(psdNumVars, ", "));

%% (6) CORRELATION + PLOTS (SOC별 1개 창, 2×nFeat subplot) ----------------
corr_tbl = table();  % SOC별로 한 행, feature별 corr_tau1/corr_tau2/N 저장

for soc = SOC_list
    Js = J(J.SOC==soc, :);

    % y 준비
    if use_log
        y_tau1 = log10(Js.tau1_s + realmin);
        y_tau2 = log10(Js.tau2_s + realmin);
        ylab1 = "log10(tau1)";
        ylab2 = "log10(tau2)";
        xPrefix = "log10(";
        xSuffix = ")";
    else
        y_tau1 = Js.tau1_s;
        y_tau2 = Js.tau2_s;
        ylab1 = "tau1 (s)";
        ylab2 = "tau2 (s)";
        xPrefix = "";
        xSuffix = "";
    end

    mTau = isfinite(y_tau1) & isfinite(y_tau2);

    row = table(soc, height(Js), 'VariableNames', {'SOC','N_total'});

    % --------- figure (SOC당 1개) ---------
    nFeat = numel(psdNumVars);
    fig = figure('Name', sprintf('SOC%d: PSD features vs tau (%s,%s)', soc, tern(use_log,"log","linear"), corr_type), ...
                 'Color','w', 'Position',[80 80 max(1200, 220*nFeat) 520]);

    % 상단: tau1, 하단: tau2
    for j = 1:nFeat
        feat = psdNumVars(j);

        % x 준비
        if use_log
            x = log10(Js.(feat) + realmin);
        else
            x = Js.(feat);
        end

        m = mTau & isfinite(x);
        Nuse = nnz(m);

        if Nuse < 3
            r1 = NaN; r2 = NaN;
        else
            r1 = corr(x(m), y_tau1(m), "Type", corr_type, "Rows","complete");
            r2 = corr(x(m), y_tau2(m), "Type", corr_type, "Rows","complete");
        end

        row.(sprintf("N_%s", feat)) = Nuse;
        row.(sprintf("corr_%s_tau1", feat)) = r1;
        row.(sprintf("corr_%s_tau2", feat)) = r2;

        % ---- plot tau1 row ----
        subplot(2, nFeat, j);
        scatter(x(m), y_tau1(m), 45, 'filled'); grid on;
        xlabel(xPrefix + feat + xSuffix, 'Interpreter','none');
        ylabel(ylab1);
        title(sprintf('%s vs tau1\nr=%.2f (N=%d)', feat, r1, Nuse), 'Interpreter','none');
        text(x(m), y_tau1(m), Js.LoadKey(m), 'FontSize',8, 'VerticalAlignment','bottom');

        % ---- plot tau2 row ----
        subplot(2, nFeat, nFeat + j);
        scatter(x(m), y_tau2(m), 45, 'filled'); grid on;
        xlabel(xPrefix + feat + xSuffix, 'Interpreter','none');
        ylabel(ylab2);
        title(sprintf('%s vs tau2\nr=%.2f (N=%d)', feat, r2, Nuse), 'Interpreter','none');
        text(x(m), y_tau2(m), Js.LoadKey(m), 'FontSize',8, 'VerticalAlignment','bottom');
    end

    sgtitle(sprintf('SOC%d | %s corr | %s', soc, corr_type, tern(use_log,"log-log","linear")), 'FontWeight','bold');

    % save fig
    tag = tern(use_log,"LOG","LIN");
    saveas(fig, fullfile(save_dir, sprintf("SOC%d_PSD_ALL_vs_tau_%s.png", soc, tag)));
    savefig(fig, fullfile(save_dir, sprintf("SOC%d_PSD_ALL_vs_tau_%s.fig", soc, tag)));

    corr_tbl = [corr_tbl; row]; %#ok<AGROW>
end

disp("==== corr_tbl (ALL features) ====");
disp(corr_tbl);

%% (7) SAVE ---------------------------------------------------------------
tag = tern(use_log,"LOG","LIN");

writetable(J,        fullfile(save_dir, sprintf("PSD_vs_ECM_join_%s.xlsx", tag)));
writetable(J,        fullfile(save_dir, sprintf("PSD_vs_ECM_join_%s.csv",  tag)));
writetable(corr_tbl, fullfile(save_dir, sprintf("corr_summary_ALL_%s.xlsx", tag)));
writetable(corr_tbl, fullfile(save_dir, sprintf("corr_summary_ALL_%s.csv",  tag)));

save(fullfile(save_dir, sprintf("PSD_vs_ECM_workspace_%s.mat", tag)), "J","corr_tbl","psd","ecm_long", ...
     "psdNumVars","use_log","corr_type","excludePSD","SOC_list");

fprintf("\n[done] saved to: %s\n", save_dir);

%% ---------------- local function: normalize load key -------------------
function key = normalizeLoadKey(loadStr)
    % Robust normalization for both PSD filenames and ECM load names.
    % Examples:
    %   "us06_0725"         -> "US06"
    %   "BSL_HW1_0725"      -> "HW1"
    %   "BSL_CITY2_0726"    -> "CITY2"
    %   "HWFET"             -> "HWFET"
    s = upper(string(loadStr));

    % 확장자 제거(안전)
    s = erase(s, ".XLSX");
    s = erase(s, ".CSV");

    % 끝 접미 숫자 제거: _0725, _0726 등
    s = regexprep(s, "_\d+$", "");

    % 대표 prefix 제거: BSL_
    s = regexprep(s, "^BSL_", "");

    % 혹시 공백/대시 변형 대응
    s = replace(s, "-", "_");
    s = replace(s, " ", "_");

    key = s;
end

function out = tern(cond, a, b)
    if cond, out = a; else, out = b; end
end
