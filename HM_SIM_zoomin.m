%% =======================================================================
%  SIM_table 일부(일부 구간 50초)만 플로팅해서 저장 (.fig 형식)
%  - 각 파일의 _SIM.mat 을 열어, 처음 N개 SIM 스텝만 0~T_WINDOW 초 구간을 그림
%  - 전압(좌 y축) / 전류(우 y축) 동시 표시
% ========================================================================

clc; clear; close all;

% [경로 설정]
base_folder = 'G:\공유 드라이브\BSL_Data4\HNE_agedcell_8_processed\SIM_parsed';
sim_mat_dir = fullfile(base_folder, '셀정렬');     % _SIM.mat 저장된 경로
out_dir     = fullfile(base_folder, 'SIM_plots_50s');  % 출력 폴더

if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% [파라미터 설정]
T_WINDOW_SEC       = 50;   % 표시할 시간 구간 (초)
MAX_SIMS_PER_FILE  = 5;   % 파일당 최대 몇 개의 SIM 스텝만 그릴지

% [대상 파일 탐색]
mats = dir(fullfile(sim_mat_dir, '*_SIM.mat'));
if isempty(mats)
    warning('SIM 테이블(.mat)을 찾지 못했습니다: %s', sim_mat_dir);
end

for f = 1:numel(mats)
    mat_path = fullfile(sim_mat_dir, mats(f).name);
    S = load(mat_path, 'SIM_table');
    if ~isfield(S, 'SIM_table')
        warning('SIM_table이 없습니다: %s', mats(f).name);
        continue;
    end
    SIM_table = S.SIM_table;

    nSIM = height(SIM_table);
    if nSIM == 0
        continue;
    end

    n_plot = min([nSIM, MAX_SIMS_PER_FILE]);
    [~, base, ~] = fileparts(mats(f).name);

    for s = 1:n_plot
        % 시간 벡터 읽기 및 상대시간(초) 변환
        t_raw = SIM_table.time{s};
        if isempty(t_raw), continue; end

        if isdatetime(t_raw)
            t_rel = seconds(t_raw - t_raw(1));
        elseif isduration(t_raw)
            t_rel = seconds(t_raw - t_raw(1));
        else
            t_rel = t_raw - t_raw(1);
        end

        % 전체 길이의 25% 지점부터 50초 구간 선택
        t0 = 0.25 * max(t_rel);
        idx = (t_rel >= t0) & (t_rel <= t0 + T_WINDOW_SEC);

        if ~any(idx)
            idx = true(size(t_rel)); % 50초보다 짧으면 전 구간
        end

        t = t_rel(idx);
        v = SIM_table.voltage{s}(idx);
        c = SIM_table.current{s}(idx);

        % 플로팅 (Visible='on' 으로 변경)
        fig = figure('Visible','on', 'Position',[200 200 900 450]);
        ax = axes(fig); hold(ax,'on');

        yyaxis left
        plot(ax, t, v, 'LineWidth', 1.3);
        ylabel(ax, 'Voltage (V)')

        yyaxis right
        plot(ax, t, c, '--', 'LineWidth', 1.2);
        ylabel(ax, 'Current (A)')

        xlabel(ax, 'Time (s)')
        title(ax, sprintf('%s | SIM %s (%.0f–%.0fs)', base, SIM_table.Properties.RowNames{s}, t0, t0+T_WINDOW_SEC))
        grid(ax, 'on')
        legend(ax, {'Voltage','Current'}, 'Location','best')

        % [저장: .fig 형식으로 저장 (saveas 사용)]
        out_name = sprintf('%s_%s_%.0fto%.0fs.fig', base, SIM_table.Properties.RowNames{s}, t0, t0+T_WINDOW_SEC);
        saveas(fig, fullfile(out_dir, out_name), 'fig');
        close(fig);
    end

    fprintf('[saved] %s → %d SIM(s) (%.0f–%.0fs) plots\n', mats(f).name, n_plot, t0, t0+T_WINDOW_SEC);
end

fprintf('\n모든 플롯 저장이 완료되었습니다.\n출력 폴더: %s\n', out_dir);
